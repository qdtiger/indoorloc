"""Export the standalone 3D scene to PNG posters and looping WebP animations."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.request import urlopen

from PIL import Image, TiffImagePlugin
import websocket

THEMES = ("light", "dark")
LANGUAGES = ("en", "zh")


@contextmanager
def scene_page(html, chromium, theme="light", width=1280, supersample=2, no_sandbox=False, lang="en"):
    """Open the scene in headless Chromium; yield a renderer from seconds to an RGB frame."""
    html = Path(html).resolve()
    if not html.is_file():
        raise FileNotFoundError(html)
    if not 720 <= width <= 2560 or supersample not in (1, 2, 3):
        raise ValueError("Use a width of 720–2560 pixels and 1–3× supersampling")
    height = round(width * 9 / 16)
    parent = None
    if "/snap/" in chromium:
        parent = Path.home() / "snap/chromium/common"
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="indoorloc-render-", dir=parent) as temporary:
        folder = Path(temporary)
        command = [chromium, "--headless", "--enable-unsafe-swiftshader", "--disable-dev-shm-usage",
                   "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0",
                   "--hide-scrollbars", "--user-data-dir=" + str(folder / "profile"), "about:blank"]
        if no_sandbox:
            command.append("--no-sandbox")
        with (folder / "chromium.log").open("wb") as log:
            browser = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        client = None
        try:
            deadline = time.monotonic() + 25
            port_file = folder / "profile/DevToolsActivePort"
            while not port_file.exists():
                if browser.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Chromium did not start: " + (folder / "chromium.log").read_text()[-1500:])
                time.sleep(.1)
            port = port_file.read_text().splitlines()[0]
            while True:
                # Some builds list extension pages first; attach to the tab itself.
                with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as response:
                    page = next((target for target in json.load(response) if target.get("type") == "page"), None)
                if page:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Chromium opened no page target")
                time.sleep(.1)
            client = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)
            sequence = 0

            def call(method, params=None):
                nonlocal sequence
                sequence += 1
                client.send(json.dumps({"id": sequence, "method": method, "params": params or {}}))
                while True:
                    result = json.loads(client.recv())
                    if result.get("id") == sequence:
                        if "error" in result:
                            raise RuntimeError(result["error"])
                        return result.get("result", {})

            def evaluate(expression):
                result = call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
                if "exceptionDetails" in result:
                    raise RuntimeError(result["exceptionDetails"].get("text", "Scene script failed"))
                return result["result"].get("value")

            call("Page.enable")
            call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height,
                                                       "deviceScaleFactor": supersample, "mobile": False})
            call("Page.navigate", {"url": html.as_uri() + f"?theme={theme}&lang={lang}"})
            deadline = time.monotonic() + 40
            while not evaluate("window.__sceneReady === true"):
                if time.monotonic() > deadline:
                    raise RuntimeError(f"The 3D scene did not load at {evaluate('location.href')}; a sandboxed "
                                       "Chromium (for example the snap) may not read files under /tmp")
                time.sleep(.2)
            if evaluate("window.__sceneMode") != "webgl":
                raise RuntimeError("Export requires a Chromium WebGL context")

            def render(seconds, full_resolution=False):
                # Let the compositor paint the updated DOM and canvas before capturing.
                evaluate(f"window.__renderScene({seconds}); new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
                shot = call("Page.captureScreenshot", {"format": "png", "optimizeForSpeed": True})
                frame = Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB")
                if not full_resolution and frame.size != (width, height):
                    frame = frame.resize((width, height), Image.Resampling.LANCZOS)
                return frame

            yield render, evaluate
        finally:
            if client:
                client.close()
            browser.terminate()
            try:
                browser.wait(timeout=8)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait()


def output_path(html, theme, suffix, lang="en"):
    html = Path(html)
    return html.with_name(html.stem + ("" if lang == "en" else f"-{lang}") + ("" if theme == "light" else f"-{theme}") + suffix)


def export_scene(html, chromium, fps=20, width=1280, themes=THEMES, supersample=2, quality=78, no_sandbox=False, langs=LANGUAGES):
    if not 1 <= fps <= 30:
        raise ValueError("Use 1–30 fps")
    outputs = []
    for lang, theme in [(lang, theme) for lang in langs for theme in themes]:
        with tempfile.TemporaryDirectory(prefix="indoorloc-frames-") as temporary:
            # Frames go into one compressed multi-page TIFF; the WebP encoder then reads one page at a time.
            stack = Path(temporary) / "frames.tif"
            writer = TiffImagePlugin.AppendingTiffWriter(stack, new=True)
            try:
                with scene_page(html, chromium, theme, width, supersample, no_sandbox, lang) as (render, evaluate):
                    count = round(evaluate("window.__sceneDuration") * fps)
                    started = time.monotonic()
                    for frame in range(count):
                        render(frame / fps).save(writer, format="TIFF", compression="tiff_deflate")
                        writer.newFrame()
                        if frame % 40 == 0 or frame == count - 1:
                            print(f"[{lang}/{theme}] rendered {frame + 1}/{count} frames ({time.monotonic() - started:.1f}s)", flush=True)
                    render(evaluate("window.__scenePosterTime"), full_resolution=True).save(output_path(html, theme, ".png", lang))
            finally:
                writer.close()
                io.BytesIO.close(writer)  # Mark it closed so garbage collection does not finalize the file twice.
            durations = [round((i + 1) * 1000 / fps) - round(i * 1000 / fps) for i in range(count)]
            destination = output_path(html, theme, ".webp", lang)
            print(f"[{lang}/{theme}] encoding WebP animation…", flush=True)
            with Image.open(stack) as frames:
                # The camera holds between moves, so one keyframe per loop keeps the still stretches nearly free.
                frames.save(destination, save_all=True, duration=durations, loop=0, quality=quality, method=6,
                            kmin=count // 2 + 1, kmax=count)
        print(f"{destination} ({destination.stat().st_size / 1024**2:.2f} MiB)")
        outputs.append(destination)
    if "light" in themes and "en" in langs:
        from examples.readme_figure import embed_animation_fallback
        embed_animation_fallback(html, output_path(html, "light", ".webp"))
    return outputs


def preview(html, chromium, seconds, directory, themes=THEMES, width=1280, supersample=2, no_sandbox=False, langs=LANGUAGES):
    """Write single frames for review, without encoding an animation."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for lang, theme in [(lang, theme) for lang in langs for theme in themes]:
        with scene_page(html, chromium, theme, width, supersample, no_sandbox, lang) as (render, _):
            for second in seconds:
                path = directory / f"{Path(html).stem}-{lang}-{theme}-{second:05.2f}s.png"
                render(second).save(path)
                print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--chromium", default=os.environ.get("CHROMIUM_BIN") or shutil.which("chromium") or shutil.which("google-chrome"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--theme", choices=(*THEMES, "both"), default="light")
    parser.add_argument("--lang", choices=(*LANGUAGES, "both"), default="both", help="Caption language(s) to export")
    parser.add_argument("--supersample", type=int, default=2, help="Render at this device scale, then downsample")
    parser.add_argument("--quality", type=int, default=78, help="Lossy WebP quality (0–100)")
    parser.add_argument("--preview", type=float, nargs="+", metavar="SECONDS", help="Only write these frames as PNG")
    parser.add_argument("--preview-dir", type=Path, default=Path("work_dirs/readme_preview"))
    parser.add_argument("--no-sandbox", action="store_true")
    arguments = parser.parse_args()
    if not arguments.chromium:
        parser.error("Install Chromium or specify --chromium /path/to/chromium")
    themes = THEMES if arguments.theme == "both" else (arguments.theme,)
    langs = LANGUAGES if arguments.lang == "both" else (arguments.lang,)
    if arguments.preview:
        preview(arguments.html, arguments.chromium, arguments.preview, arguments.preview_dir, themes,
                arguments.width, arguments.supersample, arguments.no_sandbox, langs)
    else:
        export_scene(arguments.html, arguments.chromium, arguments.fps, arguments.width, themes,
                     arguments.supersample, arguments.quality, arguments.no_sandbox, langs)
