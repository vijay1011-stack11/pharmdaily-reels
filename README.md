# pharmdaily-reels

Code-driven pipeline that turns a short pharmacology script into a narrated, captioned 1080×1920 reel for [@pharm._daily](https://instagram.com/pharm._daily), the Instagram companion to [PharmDaily](https://pharmdaily.vercel.app), a free pharmacology notes site for BDS and MBBS students.

No video editor is involved. Each reel is a script, an animated HTML scene and a render command.

> **Status:** work in progress. Six reel formats work today as separate scripts. They are being refactored into one shared core (see [Roadmap](#roadmap)).

## How it works

```
script ──► gen-voice.py ──► vo.wav + cfg.json ──► scene.html ──► render.py ──► reel.mp4
           (Kokoro TTS,      (scene cuts,          (HTML/SVG/     (headless Chromium
            caption timing)   caption timings)      Three.js)      frames → ffmpeg)
```

1. **Voice.** Each sentence goes through [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) (offline ONNX TTS). Drug names that it mispronounces are overridden with IPA.
2. **Caption timing.** No forced aligner is used. Word and phrase boundaries are first estimated from phoneme counts, then snapped to silences detected in the generated audio.
3. **Scenes.** Animations are keyed to spoken words rather than fixed timestamps, so editing the script retimes the whole video.
4. **Render.** Chromium steps through the page frame by frame. ffmpeg muxes the frames with the voice track and normalises loudness to −14 LUFS.

## Reel formats

| Format | Visual style | What it's for |
|---|---|---|
| Drug of the Day | Dark cards, chips, progress bar | One drug: mechanism, uses, adverse effects |
| Q&A Quiz | Countdown ring, answer reveal | 3 exam-style questions |
| Split | Side-by-side comparison | Two drugs, two patients |
| Blocky 3D Story | Three.js blocky characters | Story-driven explainers (e.g. grapefruit + CYP3A4) |
| Case File | Blocky 3D, MCQ scenario | Clinical case with a pause to answer |
| Wiring | Animated SVG blueprint | Analogies (e.g. general vs local anaesthesia) |

## Requirements

- Python 3.10+
- ffmpeg on `PATH`
- Kokoro model files `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) (`model-files-v1.0`). They are not committed.

```bash
pip install -r requirements.txt
playwright install chromium
```

## Roadmap

- [x] Repo setup
- [x] `core/voice.py`: shared TTS wrapper and pronunciation overrides
- [x] `core/captions.py`: one caption-timing algorithm for phrase and word captions
- [x] `core/render.py`: shared Chromium → ffmpeg renderer
- [ ] Move all six formats onto the core
- [ ] Scripts as YAML instead of Python lists
- [ ] Single CLI: `python -m reels build dotd scripts/amlodipine.yaml`
- [ ] Shared cover generator

## Content

Scripts follow KD Tripathi's *Essentials of Medical Pharmacology* and are checked by a BDS student before posting. Reels are educational and not medical advice.

## License

Code: MIT. See [LICENSE](LICENSE).
