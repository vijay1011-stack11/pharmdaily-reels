"""Shared text-to-speech layer for every reel format.

Wraps Kokoro (offline ONNX TTS) with the pieces each pipeline used to copy:
- IPA pronunciation overrides for drug names Kokoro gets wrong
- silence trimming of each generated clip
- phoneme-length weights, used later to time captions
- mixing timed clips into one voice track

Usage:
    from core.voice import Voice
    v = Voice("af_heart", speed=1.1, overrides={"Amlodipine": "æmlˈoʊdɪpiːn"})
    clip = v.say("Amlodipine, a calcium channel blocker.")
    v.write("vo.wav", [(0.14, clip)], total=clip.duration + 0.5)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 24000          # Kokoro output sample rate
LANG = "en-us"

MODEL = os.environ.get("KOKORO_MODEL", "kokoro-v1.0.onnx")
VOICES = os.environ.get("KOKORO_VOICES", "voices-v1.0.bin")

_engine = None


def engine():
    """Load Kokoro once and reuse it (loading takes a few seconds)."""
    global _engine
    if _engine is None:
        from kokoro_onnx import Kokoro
        for f in (MODEL, VOICES):
            if not Path(f).exists():
                raise FileNotFoundError(
                    f"{f} not found. Download it from github.com/thewh1teagle/kokoro-onnx "
                    "releases (model-files-v1.0), or set KOKORO_MODEL / KOKORO_VOICES.")
        _engine = Kokoro(MODEL, VOICES)
    return _engine


def db(x: float) -> float:
    """dBFS -> linear amplitude."""
    return 10 ** (x / 20)


def trim(audio: np.ndarray, threshold_db: float = -45, preroll: int = 0) -> np.ndarray:
    """Cut leading/trailing silence. `preroll` keeps a few samples before the
    first sound so soft consonants (h, f, s) aren't clipped."""
    loud = np.where(np.abs(audio) > db(threshold_db))[0]
    if len(loud) == 0:
        return audio[:0]
    return audio[max(loud[0] - preroll, 0):loud[-1] + 1]


@dataclass
class Clip:
    """One synthesised sentence."""
    text: str
    audio: np.ndarray

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


class Voice:
    def __init__(self, name: str, speed: float = 1.0,
                 overrides: dict[str, str] | None = None, preroll: int = 0):
        self.name = name
        self.speed = speed
        self.overrides = overrides or {}
        self.preroll = preroll

    def phonemes(self, text: str) -> str:
        """Phonemise text, then swap in IPA overrides for listed words."""
        tok = engine().tokenizer
        ph = tok.phonemize(text, LANG)
        for word, ipa in self.overrides.items():
            ph = ph.replace(tok.phonemize(word, LANG), ipa)
        return ph

    def weight(self, chunk: str, pause_bonus: int = 0) -> int:
        """Phoneme count of a chunk, a proxy for how long it takes to say.
        `pause_bonus` adds weight when the chunk ends in punctuation."""
        n = max(len(self.phonemes(chunk.strip(".,:?!")).replace(" ", "")), 1)
        if pause_bonus and chunk.rstrip()[-1:] in ",.:?!":
            n += pause_bonus
        return n

    def say(self, text: str) -> Clip:
        """Synthesise one sentence and trim its silence."""
        audio, _ = engine().create(self.phonemes(text), voice=self.name,
                                   speed=self.speed, lang=LANG, is_phonemes=True)
        return Clip(text, trim(audio, preroll=self.preroll))

    @staticmethod
    def mix(placed: list[tuple[float, Clip | np.ndarray]], total: float) -> np.ndarray:
        """Lay clips onto a silent track at their start times (seconds)."""
        n = int(total * SR)
        buf = np.zeros(n + SR, dtype=np.float32)
        for start, clip in placed:
            a = clip.audio if isinstance(clip, Clip) else clip
            i = int(start * SR)
            buf[i:i + len(a)] += a
        return buf[:n]

    @classmethod
    def write(cls, path: str, placed, total: float) -> None:
        import soundfile as sf
        sf.write(path, cls.mix(placed, total), SR)
