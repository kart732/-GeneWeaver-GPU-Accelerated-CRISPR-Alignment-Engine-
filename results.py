"""Hit records and TSV output."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

TSV_COLUMNS = [
    "guide", "guide_seq", "contig", "start", "end", "strand",
    "mismatches", "seed_mismatches", "mismatch_positions", "target", "pam",
]


@dataclass(frozen=True)
class Hit:
    guide: str
    guide_seq: str
    contig: str
    start: int            # 0-based, half-open, forward-strand coordinates of the protospacer
    end: int
    strand: str           # '+' or '-'
    mismatches: int
    seed_mismatches: int  # mismatches in the PAM-proximal 12 nt
    mm_positions: tuple   # 1-based positions from the guide's 5' end
    target: str           # genomic target in guide orientation; mismatches in lower case
    pam: str              # PAM in guide orientation

    @property
    def sort_key(self):
        return (self.mismatches, self.seed_mismatches, self.guide, self.contig, self.start, self.strand)

    def as_row(self) -> List[str]:
        return [
            self.guide, self.guide_seq, self.contig, str(self.start), str(self.end), self.strand,
            str(self.mismatches), str(self.seed_mismatches),
            ",".join(map(str, self.mm_positions)) or "-", self.target, self.pam,
        ]


def write_tsv(path: str, hits: Iterable[Hit]) -> int:
    rows = sorted(hits, key=lambda h: h.sort_key)
    with open(path, "w") as fh:
        fh.write("\t".join(TSV_COLUMNS) + "\n")
        for h in rows:
            fh.write("\t".join(h.as_row()) + "\n")
    return len(rows)
