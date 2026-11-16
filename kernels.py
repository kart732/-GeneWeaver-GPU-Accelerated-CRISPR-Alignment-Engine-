"""CUDA Kernel Engine (Numba).

One matching routine, ``_window_mismatches``, is compiled twice from the same
Python source:

* ``cuda.jit(device=True)`` -> runs inside the GPU kernel (JIT-compiled to PTX,
  no GIL involved),
* ``njit``                  -> CPU fallback / reference implementation.

Search model (CRISPR-Cas9 style, configurable PAM):

* strand 0 (+): protospacer = window[0:L], PAM = window[L:L+P]
* strand 1 (-): PAM (reverse-complemented) = window[0:P], protospacer
  (reverse-complemented) = window[P:P+L]; we therefore compare the *forward*
  genome bases against the reverse-complement of the guide.

Query rows are stored as ``[guide0(+), guide0(-), guide1(+), ...]`` so
``strand = query_index & 1`` and ``guide_index = query_index >> 1``.

Cheap PAM test first (rejects ~15/16 of windows for NGG), then a Hamming
distance with early exit once ``max_mismatch`` is exceeded.
"""
from __future__ import annotations

from numba import cuda, njit, uint8

MAX_Q = 64                      # queries (guide x strand) resident in shared memory per launch
MAX_L = 32                      # maximum guide length
SHARED_SIZE = MAX_Q * MAX_L     # bytes of shared memory used for the guide table
THREADS_PER_BLOCK = 256


def _window_mismatches(seq, w, guides, qoff, strand, fpam, rpam, L, P, max_mm):
    """Return mismatch count for window ``w`` vs one query, or -1 if rejected."""
    if strand == 0:
        pam_off = w + L
        proto_off = w
        for k in range(P):
            if ((1 << seq[pam_off + k]) & fpam[k]) == 0:
                return -1
    else:
        pam_off = w
        proto_off = w + P
        for k in range(P):
            if ((1 << seq[pam_off + k]) & rpam[k]) == 0:
                return -1
    mm = 0
    for j in range(L):
        if seq[proto_off + j] != guides[qoff + j]:
            mm += 1
            if mm > max_mm:
                return -1
    return mm


def _njit(**kwargs):
    """njit with on-disk caching when the cache dir is writable."""
    def deco(fn):
        try:
            return njit(cache=True, **kwargs)(fn)
        except Exception:  # pragma: no cover - read-only install location
            return njit(**kwargs)(fn)
    return deco


_core_cpu = _njit()(_window_mismatches)
_core_gpu = cuda.jit(device=True)(_window_mismatches)


@_njit(nogil=True)
def cpu_scan(seq, n_windows, guides, n_q, q_base, L, P, fpam, rpam, max_mm, hits, counter):
    """Serial reference/fallback scan. Mirrors the CUDA kernel's output contract."""
    cap = hits.shape[0]
    for w in range(n_windows):
        for q in range(n_q):
            strand = (q_base + q) & 1
            mm = _core_cpu(seq, w, guides, q * L, strand, fpam, rpam, L, P, max_mm)
            if mm >= 0:
                slot = counter[0]
                counter[0] = slot + 1
                if slot < cap:
                    hits[slot, 0] = w
                    hits[slot, 1] = q_base + q
                    hits[slot, 2] = mm


@cuda.jit
def cuda_scan(seq, n_windows, guides, n_q, q_base, L, P, fpam, rpam, max_mm, hits, counter):
    """Grid-stride kernel: thread <-> (window, query) pair.

    Consecutive threads take consecutive windows of the *same* query, so global
    reads of ``seq`` are coalesced and the guide bytes (shared memory) are a
    broadcast.  Hits are appended through a single atomic counter; ``counter``
    keeps counting past ``hits.shape[0]`` so the host can detect overflow and
    re-launch with a bigger buffer.
    """
    sh = cuda.shared.array(SHARED_SIZE, dtype=uint8)
    tid = cuda.threadIdx.x
    for k in range(tid, n_q * L, cuda.blockDim.x):
        sh[k] = guides[k]
    cuda.syncthreads()

    total = n_windows * n_q
    cap = hits.shape[0]
    for idx in range(cuda.grid(1), total, cuda.gridsize(1)):
        w = idx % n_windows
        q = idx // n_windows
        strand = (q_base + q) & 1
        mm = _core_gpu(seq, w, sh, q * L, strand, fpam, rpam, L, P, max_mm)
        if mm >= 0:
            slot = cuda.atomic.add(counter, 0, 1)
            if slot < cap:
                hits[slot, 0] = w
                hits[slot, 1] = q_base + q
                hits[slot, 2] = mm
