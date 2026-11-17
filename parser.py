"""Genomic Parser: streams a (possibly gzipped) FASTA into encoded, overlapping chunks.

Each contig is parsed with BioPython (``Bio.SeqIO``), cleansed to uint8 codes
(soft-masked lowercase -> upper, IUPAC/ambiguous -> N) and sliced into chunks of
``chunk_size`` *owned windows*.  Consecutive chunks overlap by ``window - 1``
bases so a window straddling a chunk boundary is seen by exactly one chunk.

Memory note: BioPython yields one *record* at a time, so peak host memory is
roughly 2-3x the largest contig (hg38 chr1 ~ 250 Mbp -> ~0.7 GB).
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import Iterator, List, Tuple

import numpy as np
from Bio import SeqIO

from .encoding import C, G, N, encode


@dataclass(frozen=True)
class Chunk:
    index: int
    contig: str
    contig_index: int
    offset: int          # 0-based start of ``data`` inside the contig
    data: np.ndarray     # uint8 codes, length = n_windows + window - 1
    n_windows: int       # windows *owned* by this chunk


@dataclass
class ContigInfo:
    name: str
    length: int
    n_count: int
    gc_count: int

    @property
    def gc_fraction(self) -> float:
        known = self.length - self.n_count
        return self.gc_count / known if known else 0.0


@dataclass
class ParseStats:
    contigs: List[ContigInfo] = field(default_factory=list)
    skipped_short: int = 0

    @property
    def total_bases(self) -> int:
        return sum(c.length for c in self.contigs)


def _open(path: str, mode: str):
    if str(path).endswith((".gz", ".bgz")):
        return gzip.open(path, mode)
    return open(path, mode)


def index_fasta(path: str) -> List[Tuple[str, int]]:
    """Fast pre-scan: [(contig_name, length)] without decoding sequences.

    Used only to give the dashboard a progress-bar denominator.
    """
    out: List[Tuple[str, int]] = []
    name, length = None, 0
    with _open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if name is not None:
                    out.append((name, length))
                parts = line[1:].split(None, 1)
                name = parts[0].decode("ascii", "replace") if parts else ""
                length = 0
            else:
                length += len(line.strip())
    if name is not None:
        out.append((name, length))
    return out


def count_windows(index: List[Tuple[str, int]], window: int) -> int:
    return sum(max(0, length - window + 1) for _, length in index)


class GenomicParser:
    """Iterate a FASTA file as ``Chunk`` objects ready for the GPU engine."""

    def __init__(self, path: str, window: int, chunk_size: int = 32_000_000):
        if window < 1:
            raise ValueError("window must be >= 1")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.path = path
        self.window = window
        self.chunk_size = chunk_size
        self.stats = ParseStats()

    def __iter__(self) -> Iterator[Chunk]:
        overlap = self.window - 1
        step = self.chunk_size
        chunk_index = 0
        with _open(self.path, "rt") as handle:
            for contig_index, record in enumerate(SeqIO.parse(handle, "fasta")):
                codes = encode(bytes(record.seq))
                n = len(codes)
                self.stats.contigs.append(
                    ContigInfo(
                        name=record.id,
                        length=n,
                        n_count=int((codes == N).sum()),
                        gc_count=int(((codes == C) | (codes == G)).sum()),
                    )
                )
                if n < self.window:
                    self.stats.skipped_short += 1
                    continue
                pos = 0
                while pos + self.window <= n:
                    end = min(pos + step + overlap, n)
                    data = codes[pos:end]
                    yield Chunk(
                        index=chunk_index,
                        contig=record.id,
                        contig_index=contig_index,
                        offset=pos,
                        data=data,
                        n_windows=len(data) - self.window + 1,
                    )
                    chunk_index += 1
                    pos += step
                del codes
