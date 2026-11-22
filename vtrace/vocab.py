"""Shared vocabularies and visual-confusion models.

These tables are the "domain knowledge" behind the matcher. They describe how
likely one thing is to be *observed* as another under CCTV conditions
(blur, low resolution, compression, colour casts).
"""
from __future__ import annotations

import string

import numpy as np

# ---------------------------------------------------------------------------
# Plate characters
# ---------------------------------------------------------------------------
CHARS = string.digits + string.ascii_uppercase  # 36 symbols
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
UNKNOWN = "?"

# Groups of glyphs that are frequently confused on blurred / low-res plates.
# (a, b, similarity in 0..1). Similarity is symmetric.
_CHAR_SIMILAR = [
    ("0", "O", 0.9), ("0", "D", 0.6), ("O", "D", 0.6), ("0", "Q", 0.5), ("O", "Q", 0.6),
    ("O", "U", 0.3), ("0", "U", 0.25), ("D", "B", 0.3),
    ("1", "I", 0.9), ("1", "L", 0.4), ("1", "7", 0.45), ("1", "T", 0.35), ("I", "L", 0.4),
    ("I", "T", 0.35), ("7", "T", 0.4), ("J", "1", 0.25), ("I", "J", 0.3), ("7", "Z", 0.25),
    ("2", "Z", 0.7),
    ("5", "S", 0.75), ("5", "6", 0.3), ("S", "8", 0.25), ("5", "3", 0.25),
    ("8", "B", 0.75), ("8", "3", 0.45), ("B", "3", 0.35), ("8", "0", 0.3), ("8", "6", 0.35),
    ("8", "9", 0.3), ("B", "E", 0.3), ("B", "R", 0.3), ("B", "P", 0.25),
    ("6", "G", 0.6), ("6", "C", 0.25), ("G", "C", 0.5), ("G", "O", 0.3), ("C", "O", 0.3),
    ("9", "g", 0.0), ("9", "6", 0.25), ("9", "0", 0.25), ("9", "P", 0.2),
    ("4", "A", 0.5), ("A", "H", 0.25),
    ("M", "N", 0.5), ("M", "H", 0.35), ("N", "H", 0.4), ("M", "W", 0.4), ("N", "W", 0.2),
    ("V", "Y", 0.45), ("V", "U", 0.45), ("U", "W", 0.2),
    ("E", "F", 0.6), ("P", "R", 0.55), ("P", "F", 0.3), ("K", "X", 0.45), ("K", "R", 0.25),
    ("X", "Y", 0.3), ("Q", "D", 0.35), ("Z", "S", 0.2), ("E", "3", 0.2), ("A", "R", 0.2),
]


def _char_confusion() -> np.ndarray:
    """C[k, c] = P(OCR reads symbol k | true symbol c). Columns sum to 1."""
    n = len(CHARS)
    s = np.eye(n)
    for a, b, w in _CHAR_SIMILAR:
        if a in CHAR_IDX and b in CHAR_IDX:
            s[CHAR_IDX[a], CHAR_IDX[b]] = s[CHAR_IDX[b], CHAR_IDX[a]] = w
    p_correct, p_similar_pool, p_random = 0.80, 0.16, 0.04
    c = np.zeros((n, n))
    for j in range(n):
        sim = s[:, j].copy()
        sim[j] = 0.0
        col = np.full(n, p_random / (n - 1))
        col[j] = p_correct
        if sim.sum() > 0:
            col += p_similar_pool * sim / sim.sum()
        else:
            col[j] += p_similar_pool
        c[:, j] = col / col.sum()
    return c


CHAR_CONFUSION = _char_confusion()


def char_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    for x, y, w in _CHAR_SIMILAR:
        if {x, y} == {a, b}:
            return w
    return 0.0


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOURS = ["white", "silver", "grey", "black", "red", "maroon", "blue", "green",
           "yellow", "orange", "brown", "beige"]

# Reference colours (RGB) used to name pixel clusters.
COLOUR_RGB = {
    "white": (235, 235, 235), "silver": (185, 188, 192), "grey": (115, 118, 122),
    "black": (25, 25, 28), "red": (190, 25, 30), "maroon": (110, 20, 30),
    "blue": (30, 70, 170), "green": (40, 130, 60), "yellow": (230, 200, 30),
    "orange": (235, 120, 25), "brown": (110, 70, 40), "beige": (210, 190, 150),
}

_COLOUR_SIMILAR = [
    ("white", "silver", 0.55), ("silver", "grey", 0.6), ("grey", "black", 0.35),
    ("white", "beige", 0.4), ("silver", "beige", 0.25), ("red", "maroon", 0.55),
    ("red", "orange", 0.35), ("orange", "yellow", 0.4), ("maroon", "brown", 0.45),
    ("brown", "beige", 0.35), ("brown", "orange", 0.25), ("blue", "grey", 0.2),
    ("blue", "black", 0.15), ("green", "grey", 0.12), ("maroon", "black", 0.2),
    ("white", "grey", 0.2), ("yellow", "beige", 0.25), ("green", "yellow", 0.15),
]

# ---------------------------------------------------------------------------
# Body types
# ---------------------------------------------------------------------------
BODY_TYPES = ["car", "suv", "van", "two-wheeler", "auto-rickshaw", "truck", "bus"]

_BODY_SIMILAR = [
    ("car", "suv", 0.45), ("car", "van", 0.3), ("suv", "van", 0.45), ("truck", "bus", 0.3),
    ("van", "truck", 0.25), ("auto-rickshaw", "two-wheeler", 0.15),
    ("auto-rickshaw", "car", 0.1), ("van", "bus", 0.2), ("suv", "truck", 0.15),
]


def _confusion(labels, similar, p_correct, p_similar_pool, p_random):
    n = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    s = np.eye(n)
    for a, b, w in similar:
        s[idx[a], idx[b]] = s[idx[b], idx[a]] = w
    c = np.zeros((n, n))
    for j in range(n):
        sim = s[:, j].copy()
        sim[j] = 0
        col = np.full(n, p_random / (n - 1))
        col[j] = p_correct
        col += p_similar_pool * (sim / sim.sum() if sim.sum() else np.eye(n)[j])
        c[:, j] = col / col.sum()
    return c


COLOUR_CONFUSION = _confusion(COLOURS, _COLOUR_SIMILAR, 0.62, 0.28, 0.10)
BODY_CONFUSION = _confusion(BODY_TYPES, _BODY_SIMILAR, 0.72, 0.20, 0.08)


def similar_label(a: str, b: str, table) -> float:
    if a == b:
        return 1.0
    for x, y, w in table:
        if {x, y} == {a, b}:
            return w
    return 0.0


def colour_similarity(a, b):
    return similar_label(a, b, _COLOUR_SIMILAR)


def body_similarity(a, b):
    return similar_label(a, b, _BODY_SIMILAR)


# Indian plate grammar. L = letter, D = digit.
# Standard:  SS DD L{0,3} DDDD   e.g. MH12DE4521, DL3CAF0912 (DL uses 1-2 digit RTO)
# BH series: DD BH DDDD L{1,2}   e.g. 22BH1234AA
INDIAN_STATE_CODES = [
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "JH",
    "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PY", "PB", "RJ", "SK",
    "TN", "TS", "TR", "UP", "UK", "WB",
]
