"""Distributed Scheduler (Dask).

The parser (running in the driver process) produces chunks lazily.  At most
``inflight`` chunks are submitted to the Dask cluster at any time (back-pressure,
so a 3 Gbp genome never sits in RAM at once).  Each worker process owns one
device: workers are numbered 0..N-1 and worker *i* binds to GPU ``i % n_gpus``.

Modes
-----
* ``dask``  - LocalCluster (one single-threaded process per GPU, or per CPU core
              on the fallback path), or an existing cluster via ``dask_address``.
* ``sync``  - everything in-process; used for tests, debugging and tiny inputs.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .engine import AlignmentEngine, QuerySpec, detect_backend
from .parser import ContigInfo, GenomicParser, count_windows, index_fasta
from .results import Hit
from .telemetry import gpu_count


@dataclass
class ScanConfig:
    fasta: str
    spec: QuerySpec
    chunk_size: int = 32_000_000
    backend: str = "auto"              # auto | cuda | cpu
    mode: str = "dask"                 # dask | sync
    n_workers: Optional[int] = None
    dask_address: Optional[str] = None
    max_hits_per_chunk: int = 1 << 20
    inflight: Optional[int] = None
    skip_index: bool = False


@dataclass
class ChunkResult:
    chunk_index: int
    contig: str
    offset: int
    n_windows: int
    hits: List[Hit]
    total_hits: int
    seconds: float
    kernel_seconds: float
    device: str
    worker: str
    resized: bool = False


@dataclass
class ScanSummary:
    windows_scanned: int = 0
    chunks: int = 0
    total_hits: int = 0
    seconds: float = 0.0
    kernel_seconds: float = 0.0
    backend: str = ""
    hits: List[Hit] = field(default_factory=list)
    contigs: List[ContigInfo] = field(default_factory=list)
    cancelled: bool = False

    @property
    def throughput(self) -> float:
        """Genome windows (bases) scanned per second, wall clock."""
        return self.windows_scanned / self.seconds if self.seconds else 0.0


# --------------------------------------------------------------- worker side
_ENGINES: Dict[tuple, AlignmentEngine] = {}


def _worker_identity():
    """(numeric worker index, label) for the current Dask worker, or (0, 'local')."""
    try:
        from distributed import get_worker
        w = get_worker()
        name = w.name
        if isinstance(name, int) or str(name).isdigit():
            idx = int(name)
        else:
            idx = zlib.crc32(str(name).encode())
        return idx, f"worker-{name}"
    except Exception:
        return 0, "local"


def _get_engine(spec: QuerySpec, backend: str, max_hits: int, worker_idx: int) -> AlignmentEngine:
    key = (spec.signature(), backend, max_hits, worker_idx)
    eng = _ENGINES.get(key)
    if eng is None:
        device = 0
        if backend == "cuda":
            from numba import cuda
            device = worker_idx % max(1, len(cuda.gpus))
        eng = AlignmentEngine(spec, backend=backend, device=device, max_hits=max_hits)
        _ENGINES[key] = eng
    return eng


def process_chunk(meta: dict, data: np.ndarray, spec: QuerySpec, backend: str,
                  max_hits: int) -> ChunkResult:
    """Runs on a Dask worker: scan one chunk and decode its hits."""
    t0 = time.perf_counter()
    worker_idx, label = _worker_identity()
    eng = _get_engine(spec, backend, max_hits, worker_idx)
    out = eng.scan(data, meta["n_windows"])
    hits = eng.decode_hits(out.hits, data, meta["contig"], meta["offset"]) if len(out.hits) else []
    return ChunkResult(
        chunk_index=meta["index"], contig=meta["contig"], offset=meta["offset"],
        n_windows=meta["n_windows"], hits=hits, total_hits=out.total_hits,
        seconds=time.perf_counter() - t0, kernel_seconds=out.kernel_seconds,
        device=eng.device_name, worker=label, resized=out.resized,
    )


# --------------------------------------------------------------- driver side
Callback = Callable[[str, object], None]


def _meta(chunk) -> dict:
    return {"index": chunk.index, "contig": chunk.contig, "offset": chunk.offset,
            "n_windows": chunk.n_windows}


def run_scan(cfg: ScanConfig, on_event: Optional[Callback] = None,
             stop: Optional[threading.Event] = None) -> ScanSummary:
    """Run a full scan.  Events: 'start' (dict), 'chunk' (ChunkResult), 'done' (ScanSummary)."""
    emit = on_event or (lambda *_: None)
    stop = stop or threading.Event()
    spec = cfg.spec
    backend = detect_backend(cfg.backend)

    total_windows = None
    if not cfg.skip_index:
        emit("status", "Indexing FASTA ...")
        idx = index_fasta(cfg.fasta)
        total_windows = count_windows(idx, spec.window)
        emit("start", {"total_windows": total_windows, "contigs": len(idx), "backend": backend})
    else:
        emit("start", {"total_windows": None, "contigs": None, "backend": backend})

    parser = GenomicParser(cfg.fasta, spec.window, cfg.chunk_size)
    summary = ScanSummary(backend=backend)
    t_start = time.perf_counter()

    def record(res: ChunkResult):
        summary.chunks += 1
        summary.windows_scanned += res.n_windows
        summary.total_hits += res.total_hits
        summary.kernel_seconds += res.kernel_seconds
        summary.hits.extend(res.hits)
        emit("chunk", res)

    if cfg.mode == "sync":
        for chunk in parser:
            if stop.is_set():
                summary.cancelled = True
                break
            record(process_chunk(_meta(chunk), chunk.data, spec, backend, cfg.max_hits_per_chunk))
    elif cfg.mode == "dask":
        _run_dask(cfg, parser, spec, backend, record, stop, summary, emit)
    else:
        raise ValueError(f"unknown scheduler mode {cfg.mode!r}")

    summary.seconds = time.perf_counter() - t_start
    summary.contigs = parser.stats.contigs
    emit("done", summary)
    return summary


def start_local_cluster(backend: str, n_workers: Optional[int] = None):
    """One single-threaded worker process per GPU (or per CPU core on the fallback path).

    Call this *before* launching the Textual dashboard: Textual takes over the
    terminal file descriptors, which breaks spawning worker processes afterwards.
    """
    from distributed import LocalCluster
    backend = detect_backend(backend)
    if n_workers:
        n = n_workers
    elif backend == "cuda":
        n = max(1, gpu_count())
    else:
        n = os.cpu_count() or 1
    return LocalCluster(n_workers=n, threads_per_worker=1, processes=True,
                        dashboard_address=None, silence_logs=logging.ERROR)


def _run_dask(cfg, parser, spec, backend, record, stop, summary, emit):
    from distributed import Client, as_completed

    emit("status", "Starting Dask cluster ...")
    cluster = None
    if cfg.dask_address:
        client = Client(cfg.dask_address)
    else:
        cluster = start_local_cluster(backend, cfg.n_workers)
        client = Client(cluster)
    try:
        n_workers = max(1, len(client.scheduler_info()["workers"]))
        emit("status", f"Dask: {n_workers} worker(s), backend={backend}")
        inflight = cfg.inflight or 2 * n_workers
        chunks = iter(parser)
        ac = as_completed()
        held = {}

        def submit_next() -> bool:
            try:
                chunk = next(chunks)
            except StopIteration:
                return False
            data_f = client.scatter(chunk.data, hash=False)
            fut = client.submit(process_chunk, _meta(chunk), data_f, spec, backend,
                                cfg.max_hits_per_chunk, pure=False)
            held[fut.key] = data_f
            ac.add(fut)
            return True

        for _ in range(inflight):
            if stop.is_set() or not submit_next():
                break
        for fut in ac:
            held.pop(fut.key, None)
            record(fut.result())
            if stop.is_set():
                summary.cancelled = True
                continue          # drain what is already in flight, submit nothing new
            submit_next()
    finally:
        client.close()
        if cluster is not None:
            cluster.close()
