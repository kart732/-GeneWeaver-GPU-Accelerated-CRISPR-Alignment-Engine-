"""Nucleotide encoding helpers.

Bases are stored as uint8 codes: A=0, C=1, G=2, T=3, anything else (N, IUPAC
ambiguity codes, gaps) = 4.  One byte per base keeps host->device copies simple;
2-bit packing is a natural next optimisation (see README).

PAM motifs are expressed as per-position bitmasks over {A=1, C=2, G=4, T=8}
(IUPAC), so a base code ``c`` matches mask ``m`` iff ``(1 << c) & m != 0``.
Code 4 (N) gives bit 16, which never overlaps a 4-bit mask, so an ambiguous
genomic base can never satisfy a PAM.
"""
from __future__ import annotations

import numpy as np

A, C, G, T, N = 0, 1, 2, 3, 4

_ENCODE_LUT = np.full(256, N, dtype=np.uint8)
for _i, _ch in enumerate("ACGT"):
    _ENCODE_LUT[ord(_ch)] = _i
    _ENCODE_LUT[ord(_ch.lower())] = _i

_DECODE_LUT = np.frombuffer(b"ACGTN", dtype=np.uint8)

IUPAC_MASKS = {
    "A": 0b0001, "C": 0b0010, "G": 0b0100, "T": 0b1000,
    "R": 0b0101, "Y": 0b1010, "S": 0b0110, "W": 0b1001,
    "K": 0b1100, "M": 0b0011, "B": 0b1110, "D": 0b1101,
    "H": 0b1011, "V": 0b0111, "N": 0b1111,
}


def encode(seq) -> np.ndarray:
    """str / bytes / bytearray -> writable uint8 code array (case-insensitive)."""
    if isinstance(seq, str):
        seq = seq.encode("ascii", "replace")
    raw = np.frombuffer(seq, dtype=np.uint8)
    return _ENCODE_LUT[raw]


def decode(codes: np.ndarray) -> str:
    """uint8 code array -> upper-case string."""
    return _DECODE_LUT[np.minimum(codes, N)].tobytes().decode("ascii")


def revcomp_codes(codes: np.ndarray) -> np.ndarray:
    """Reverse-complement of a code array (N stays N)."""
    out = codes[::-1].copy()
    m = out < N
    out[m] = 3 - out[m]
    return out


def revcomp_str(seq: str) -> str:
    return decode(revcomp_codes(encode(seq)))


def _complement_mask(m: int) -> int:
    return ((m & 1) << 3) | ((m & 8) >> 3) | ((m & 2) << 1) | ((m & 4) >> 1)


def pam_masks(pam: str) -> np.ndarray:
    """'NGG' -> uint8 array of IUPAC bitmasks, one per PAM position."""
    try:
        return np.array([IUPAC_MASKS[ch] for ch in pam.upper()], dtype=np.uint8)
    except KeyError as exc:
        raise ValueError(f"Invalid IUPAC character {exc} in PAM {pam!r}") from None


def revcomp_masks(masks: np.ndarray) -> np.ndarray:
    """PAM masks as they appear on the opposite strand (reverse + complement)."""
    return np.array([_complement_mask(int(m)) for m in masks[::-1]], dtype=np.uint8)
