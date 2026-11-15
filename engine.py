"""Alignment engine: owns the guide table, device buffers and kernel launches."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import encoding as enc
from .kernels import MAX_L, MAX_Q, THREADS_PER_BLOCK
from .results import Hit

SEED_LENGTH = 12


def cudasim_enabled() -> bool:
    from numba.core import config
    return bool(config.ENABLE_CUDASIM)


def detect_backend(preferred: str = "auto") -> str:
    """Return 'cuda' or 'cpu'.  (The CUDA simulator counts as 'cuda'.)"""
    if preferred not in ("auto", "cuda", "cpu"):
        raise ValueError(f"unknown backend {preferred!r}")
    if preferred == "cpu":
        return "cpu"
    from numba import cuda
    try:
        if cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if preferred == "cuda":
        raise RuntimeError(
            "backend 'cuda' requested but no usable NVIDIA GPU/driver was found "
            "(use --backend cpu, or --cudasim to run the kernel in Numba's simulator)"
        )
    return "cpu"


@dataclass
class QuerySpec:
    """Guides + search parameters, pre-encoded for the kernels."""
    names: List[str]
    guides: List[str]
    max_mismatch: int = 4
    pam: str = "NGG"

    L: int = field(init=False)
    P: int = field(init=False)
    guide_codes: np.ndarray = field(init=False, repr=False)   # (G, L)
    queries: np.ndarray = field(init=False, repr=False)       # (2G, L) fwd/rc interleaved
    fpam: np.ndarray = field(init=False, repr=False)
    rpam: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not self.guides:
            raise ValueError("at least one guide sequence is required")
        if len(self.names) != len(self.guides):
            raise ValueError("names and guides must have the same length")
        self.guides = [g.strip().upper() for g in self.guides]
        lengths = {len(g) for g in self.guides}
        if len(lengths) != 1:
            raise ValueError(f"all guides must have the same length, got {sorted(lengths)}")
        self.L = lengths.pop()
        if not 1 <= self.L <= MAX_L:
            raise ValueError(f"guide length must be 1..{MAX_L}, got {self.L}")
        for n, g in zip(self.names, self.guides):
            bad = set(g) - set("ACGT")
            if bad:
                raise ValueError(f"guide {n!r} contains non-ACGT characters: {sorted(bad)}")
        if not 0 <= self.max_mismatch <= self.L:
            raise ValueError(f"max_mismatch must be 0..{self.L}")
        self.pam = self.pam.upper()
        self.fpam = enc.pam_masks(self.pam) if self.pam else np.zeros(0, np.uint8)
        self.rpam = enc.revcomp_masks(self.fpam)
        self.P = len(self.fpam)
        self.guide_codes = np.stack([enc.encode(g) for g in self.guides])
        rows = []
        for gc in self.guide_codes:
            rows.append(gc)
            rows.append(enc.revcomp_codes(gc))
        self.queries = np.ascontiguousarray(np.stack(rows), dtype=np.uint8)

    @property
    def window(self) -> int:
        return self.L + self.P

    @property
    def n_queries(self) -> int:
        return self.queries.shape[0]

    def signature(self):
        return (tuple(self.guides), self.pam, self.max_mismatch)


@dataclass
class ScanOutput:
    hits: np.ndarray          # (n, 3) int32: local window start, query index, mismatches
    total_hits: int
    kernel_seconds: float
    resized: bool             # hit buffer had to be enlarged and the batch re-run


class AlignmentEngine:
    """Runs the scan on one device (a single GPU, or the CPU fallback)."""

    def __init__(self, spec: QuerySpec, backend: str = "auto", device: int = 0,
                 max_hits: int = 1 << 20):
        self.spec = spec
        self.backend = detect_backend(backend)
        self.device_index = device
        self.max_hits = max_hits
        self.device_name = "CPU (numba njit)"

        if self.backend == "cuda":
            from numba import cuda
            cuda.select_device(device)
            name, self._sm_count = "GPU", 16
            try:  # the CUDA simulator has no get_current_device()
                dev = cuda.get_current_device()
                raw = getattr(dev, "name", b"GPU")
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
                self._sm_count = int(dev.MULTIPROCESSOR_COUNT)
            except Exception:
                pass
            self.device_name = f"{name} #{device}"
            if cudasim_enabled():
                self.device_name = "CUDA simulator"
            self._d_fpam = cuda.to_device(spec.fpam)
            self._d_rpam = cuda.to_device(spec.rpam)
            # Guide table split in batches of MAX_Q queries (the shared-memory capacity).
            self._batches = []
            for b0 in range(0, spec.n_queries, MAX_Q):
                rows = spec.queries[b0:b0 + MAX_Q]
                self._batches.append((b0, len(rows), cuda.to_device(np.ascontiguousarray(rows.ravel()))))
            self._d_hits = cuda.device_array((max_hits, 3), dtype=np.int32)
            self._d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))
        else:
            self._flat_queries = np.ascontiguousarray(spec.queries.ravel())
            self._hits = np.empty((max_hits, 3), dtype=np.int32)

    # ------------------------------------------------------------------ scan
    def scan(self, seq: np.ndarray, n_windows: int) -> ScanOutput:
        if n_windows <= 0:
            return ScanOutput(np.empty((0, 3), np.int32), 0, 0.0, False)
        if self.backend == "cuda":
            return self._scan_cuda(seq, n_windows)
        return self._scan_cpu(seq, n_windows)

    def _scan_cpu(self, seq, n_windows) -> ScanOutput:
        from .kernels import cpu_scan
        s = self.spec
        counter = np.zeros(1, dtype=np.int64)
        resized = False
        t0 = time.perf_counter()
        cpu_scan(seq, n_windows, self._flat_queries, s.n_queries, 0, s.L, s.P,
                 s.fpam, s.rpam, s.max_mismatch, self._hits, counter)
        total = int(counter[0])
        if total > self._hits.shape[0]:
            resized = True
            self._hits = np.empty((total + total // 4 + 1024, 3), dtype=np.int32)
            counter[0] = 0
            cpu_scan(seq, n_windows, self._flat_queries, s.n_queries, 0, s.L, s.P,
                     s.fpam, s.rpam, s.max_mismatch, self._hits, counter)
        dt = time.perf_counter() - t0
        return ScanOutput(self._hits[:total].copy(), total, dt, resized)

    def _scan_cuda(self, seq, n_windows) -> ScanOutput:
        from numba import cuda
        from .kernels import cuda_scan
        s = self.spec
        d_seq = cuda.to_device(seq)
        outs, total_hits, kernel_s, resized = [], 0, 0.0, False
        for q_base, n_q, d_guides in self._batches:
            total = n_windows * n_q
            blocks = int(max(1, min((total + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
                                    self._sm_count * 32)))
            for attempt in range(2):
                self._d_counter.copy_to_device(np.zeros(1, dtype=np.int32))
                t0 = time.perf_counter()
                cuda_scan[blocks, THREADS_PER_BLOCK](
                    d_seq, n_windows, d_guides, n_q, q_base, s.L, s.P,
                    self._d_fpam, self._d_rpam, s.max_mismatch, self._d_hits, self._d_counter)
                cuda.synchronize()
                kernel_s += time.perf_counter() - t0
                count = int(self._d_counter.copy_to_host()[0])
                if count <= self._d_hits.shape[0] or attempt == 1:
                    break
                # Hit buffer overflowed: grow it once to the exact need and re-run this batch.
                resized = True
                self.max_hits = count + count // 4 + 1024
                self._d_hits = cuda.device_array((self.max_hits, 3), dtype=np.int32)
            n_keep = min(count, self._d_hits.shape[0])
            if n_keep:
                outs.append(self._d_hits[:n_keep].copy_to_host())
            total_hits += count
        hits = np.concatenate(outs) if outs else np.empty((0, 3), np.int32)
        return ScanOutput(hits, total_hits, kernel_s, resized)

    # ---------------------------------------------------------------- decode
    def decode_hits(self, raw: np.ndarray, seq: np.ndarray, contig: str, offset: int) -> List[Hit]:
        """Turn raw (window, query, mm) rows into annotated Hit records."""
        s = self.spec
        L, P = s.L, s.P
        out: List[Hit] = []
        for w, qidx, mm in raw.tolist():
            g, strand = qidx >> 1, qidx & 1
            if strand == 0:
                proto = seq[w:w + L]
                pam = seq[w + L:w + L + P]
                start = offset + w
            else:
                proto = enc.revcomp_codes(seq[w + P:w + P + L])
                pam = enc.revcomp_codes(seq[w:w + P])
                start = offset + w + P
            gcodes = s.guide_codes[g]
            diff = proto != gcodes
            positions = tuple((np.nonzero(diff)[0] + 1).tolist())
            letters = enc.decode(proto)
            target = "".join(ch.lower() if d else ch for ch, d in zip(letters, diff.tolist()))
            seed = sum(1 for p in positions if p > L - SEED_LENGTH)
            out.append(Hit(
                guide=s.names[g], guide_seq=s.guides[g], contig=contig,
                start=start, end=start + L, strand="+" if strand == 0 else "-",
                mismatches=int(mm), seed_mismatches=seed, mm_positions=positions,
                target=target, pam=enc.decode(pam),
            ))
        return out
