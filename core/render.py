"""Shared renderer: animated HTML scene + cfg.json + vo.wav -> 1080x1920 mp4.

Every reel format draws itself in a browser page that exposes one function,
`render(t)` (or `renderAt(t)`), which paints the frame for time t in seconds.
This module steps Chromium through those frames and pipes the screenshots
into ffmpeg together with the voice track.

What it unifies (each format used to carry its own copy):
- serves the scene folder over a local HTTP server, so ES modules / Three.js
  work without running `python -m http.server` by hand
- injects cfg.json as `window.CFG` before the page loads
- waits for fonts, and for `window.READY === true` if the page sets it
- optional WebGL via SwiftShader for the Three.js formats (works without a GPU)
- one ffmpeg setting for every format: H.264, AAC 48 kHz, loudness -14 LUFS

CLI:
    python -m core.render scene.html                  # -> out.mp4
    python -m core.render scene.html -o reel.mp4 --webgl
    python -m core.render scene.html --stills 1,5.5,12  # PNGs for a quick check
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

W, H, FPS = 1080, 1920, 30

WEBGL_ARGS = ["--use-gl=angle", "--use-angle=swiftshader",
              "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(folder: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    """Serve `folder` on a free localhost port in a background thread."""
    handler = functools.partial(_Quiet, directory=str(folder))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def ffmpeg_cmd(audio: Path, out: Path, fps: int = FPS) -> list[str]:
    return ["ffmpeg", "-v", "error", "-y",
            "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
            "-i", str(audio),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", "-movflags", "+faststart", str(out)]


async def _render(scene: Path, cfg: dict, audio: Path, out: Path, *,
                  stills: list[float] | None, webgl: bool, fps: int) -> None:
    from playwright.async_api import async_playwright

    srv, port = serve(scene.parent)
    args = ["--force-device-scale-factor=1"] + (WEBGL_ARGS if webgl else [])
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                executable_path=os.environ.get("CHROME_PATH") or None, args=args)
            page = await browser.new_page(viewport={"width": W, "height": H})
            await page.add_init_script(f"window.CFG={json.dumps(cfg, ensure_ascii=False)};")
            await page.goto(f"http://127.0.0.1:{port}/{scene.name}")
            await page.evaluate("document.fonts.ready")
            if await page.evaluate("'READY' in window"):
                await page.wait_for_function("window.READY === true", timeout=60000)
            fn = await page.evaluate(
                "typeof window.render === 'function' ? 'render' : "
                "typeof window.renderAt === 'function' ? 'renderAt' : ''")
            if not fn:
                raise RuntimeError(f"{scene.name} must define window.render(t) or window.renderAt(t)")

            if stills:
                for t in stills:
                    await page.evaluate(f"{fn}({t})")
                    await page.screenshot(path=str(out.parent / f"still_{t}.png"))
                print(f"{len(stills)} stills -> {out.parent}")
                return

            n = int(cfg["T"] * fps)
            ff = subprocess.Popen(ffmpeg_cmd(audio, out, fps), stdin=subprocess.PIPE)
            t0 = time.time()
            for i in range(n):
                await page.evaluate(f"{fn}({i / fps})")   # awaits a returned Promise too
                ff.stdin.write(await page.screenshot(type="jpeg", quality=95))
                if i and i % (fps * 2) == 0:
                    rate = (time.time() - t0) / i
                    print(f"\r{i}/{n} frames, ~{rate * (n - i):.0f}s left", end="", flush=True)
            ff.stdin.close()
            if ff.wait() != 0:
                raise RuntimeError("ffmpeg failed")
            print(f"\r{n}/{n} frames in {time.time() - t0:.0f}s -> {out}")
    finally:
        srv.shutdown()


def render(scene: str | Path, out: str | Path = "out.mp4", *,
           cfg: str | Path | None = None, audio: str | Path | None = None,
           stills: list[float] | None = None, webgl: bool = False, fps: int = FPS) -> None:
    """Render `scene` to `out`. cfg.json and vo.wav default to the scene's folder."""
    scene = Path(scene).resolve()
    cfg_path = Path(cfg) if cfg else scene.parent / "cfg.json"
    audio = Path(audio) if audio else scene.parent / "vo.wav"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    asyncio.run(_render(scene, data, audio, Path(out).resolve(),
                        stills=stills, webgl=webgl, fps=fps))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Render a reel scene to mp4.")
    ap.add_argument("scene", help="scene HTML file")
    ap.add_argument("-o", "--out", default="out.mp4")
    ap.add_argument("--cfg", help="timing file (default: cfg.json next to the scene)")
    ap.add_argument("--audio", help="voice track (default: vo.wav next to the scene)")
    ap.add_argument("--stills", help="comma-separated times; saves PNGs instead of a video")
    ap.add_argument("--webgl", action="store_true", help="enable WebGL (Three.js scenes)")
    ap.add_argument("--fps", type=int, default=FPS)
    a = ap.parse_args(argv)
    render(a.scene, a.out, cfg=a.cfg, audio=a.audio, webgl=a.webgl, fps=a.fps,
           stills=[float(x) for x in a.stills.split(",")] if a.stills else None)


if __name__ == "__main__":
    sys.exit(main())
