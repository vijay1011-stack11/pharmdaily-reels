"""Caption timing without a forced aligner.

Each sentence is synthesised as one clip, so we know where it starts and ends
but not where each word or phrase falls inside it. We estimate that in 3 steps:

1. Estimate: split the clip's duration in proportion to each chunk's phoneme
   count (longer to say = more time). Chunks ending in punctuation get extra
   weight for the pause after them.
2. Snap: Kokoro leaves short silences at commas and full stops. Detect them
   and move nearby boundaries onto them. Boundaries after punctuation may
   move further than others, because a pause there is expected.
3. Re-spread: the boundaries that weren't snapped are re-spread in proportion
   between the snapped ones, so one good anchor corrects its neighbours too.

Works for both caption styles:
- phrases (2-4 words per caption, DOTD / Q&A / Split)
- single words (Blocky / Case File / Wiring)

Before this module the two styles used two slightly different copies of this
logic. This is the merged version.
"""
from __future__ import annotations

import re

import numpy as np

from .voice import SR, Clip, Voice, db

PUNCT = ",.:?!;"

# A chunk is (spoken, display): `spoken` goes to TTS, `display` is shown
# (may contain *teal* markup or a different spelling, e.g. "C Y P 3A4" -> "CYP3A4").
Chunk = tuple[str, str]


def silences(audio: np.ndarray, threshold_db: float = -37, min_len: float = 0.08) -> list[float]:
    """Midpoints (seconds) of the gaps inside a clip that are at least `min_len` long."""
    win = int(0.01 * SR)
    loud = np.convolve((np.abs(audio) > db(threshold_db)).astype(float),
                       np.ones(win) / win, "same") > 0.02
    out, i, n = [], 0, len(loud)
    while i < n:
        if loud[i]:
            i += 1
            continue
        j = i
        while j < n and not loud[j]:
            j += 1
        if (j - i) / SR >= min_len and i > 0 and j < n:   # inner gaps only
            out.append((i + j) / 2 / SR)
        i = j
    return out


def time_chunks(clip: Clip, chunks: list[Chunk], voice: Voice, *,
                snap: float = 0.25, snap_punct: float = 0.5,
                min_gap: float = 0.3, pause_bonus: int = 4) -> list[float]:
    """Return len(chunks)+1 edge times (seconds, relative to the clip start).

    snap        how far a mid-sentence boundary may move to reach a silence
                (use 0 for single words: there's rarely a pause between them)
    snap_punct  how far a boundary after punctuation may move
    min_gap     shortest time any chunk is allowed on screen
    """
    dur = clip.duration
    n = len(chunks)
    if n == 1:
        return [0.0, dur]

    # 1. estimate from phoneme weights
    w = [voice.weight(c[0], pause_bonus) for c in chunks]
    total = sum(w)
    est = list(np.cumsum(w)[:-1] / total * dur)

    # 2. snap boundaries to detected silences
    gaps = silences(clip.audio)
    anchors = [(-1, 0.0)]                      # (boundary index, time); -1 = clip start
    used: set[float] = set()
    for j, t in enumerate(est):
        reach = snap_punct if chunks[j][0].rstrip()[-1:] in PUNCT else snap
        if reach <= 0:
            continue
        lo = anchors[-1][1] + min_gap
        cands = [g for g in gaps if g not in used and g > lo and abs(g - t) <= reach]
        if cands:
            g = min(cands, key=lambda g: abs(g - t))
            used.add(g)
            anchors.append((j, g))
    anchors.append((n - 1, dur))               # clip end

    # 3. re-spread the unsnapped boundaries between anchors
    edges = [0.0]
    for (j0, t0), (j1, t1) in zip(anchors, anchors[1:]):
        seg = w[j0 + 1:j1 + 1]
        acc = 0
        for x in seg[:-1]:
            acc += x
            edges.append(t0 + acc / sum(seg) * (t1 - t0))
        edges.append(t1)

    # keep every chunk on screen for at least min_gap (bounded by the clip)
    for k in range(1, n):
        edges[k] = min(max(edges[k], edges[k - 1] + min_gap), dur)
    return edges


def parse_words(line: str) -> list[Chunk]:
    """Word-level script syntax: plain words, or [spoken|display] for a custom caption.
    "Easy trick: amides have [two eyes,|two i's]" ->
    [("Easy","Easy"), ("trick:","trick"), ..., ("two eyes,","two i's")]
    """
    out = []
    for m in re.finditer(r"\[([^|]+)\|([^\]]+)\]|(\S+)", line):
        if m.group(3):
            out.append((m.group(3), m.group(3).strip(PUNCT)))
        else:
            out.append((m.group(1), m.group(2)))
    return out


def phrases(*items) -> list[Chunk]:
    """Phrase-level script syntax: "text" or ("spoken", "*display*")."""
    return [(x, x) if isinstance(x, str) else tuple(x) for x in items]
