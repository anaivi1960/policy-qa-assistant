"""Record a captioned demo video of the service from a REAL live session.

    python demo/make_demo.py            # needs ffmpeg on PATH and ANTHROPIC_API_KEY in .env

What it does:
1. Starts the service (`python run.py`) and captures its real startup log.
2. Runs each demo command (the curl examples from the README, plus a few more) in bash
   against the live service and captures the real output.
3. Renders the captured session as a terminal animation with burned-in captions, and
   writes demo/demo.mp4 plus demo/demo.srt (the same captions as a subtitle file).

Nothing shown in the video is scripted output: every response is what the service
returned during the recording run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 15
BG, FG, DIM, GREEN, BLUE, CAP_BG = (24, 26, 33), (220, 223, 228), (130, 136, 150), (120, 200, 120), (110, 170, 255), (0, 0, 0)
MONO = ImageFont.truetype(os.getenv("DEMO_MONO_FONT", "C:/Windows/Fonts/consola.ttf"), 17)
CAP = ImageFont.truetype(os.getenv("DEMO_CAPTION_FONT", "C:/Windows/Fonts/segoeui.ttf"), 26)
TITLE = ImageFont.truetype(os.getenv("DEMO_CAPTION_FONT", "C:/Windows/Fonts/segoeuib.ttf"), 44)
LINE_H, COLS, TOP, LEFT = 22, 112, 54, 28
MAX_LINES = (H - TOP - 110) // LINE_H
PROMPT = "policy-qa-assistant $ "
BASH = shutil.which("bash") or "bash"

ASK = 'curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d \'{"question": "%s"}\' | python -m json.tool'

SCENES = [
    ("Check the service is up: GET /health", "curl -s http://localhost:8000/health"),
    ("Ask a question (README example). Each quote is verified word for word, and each sentence is checked against the quotes.",
     ASK % "How much paternity leave do I get?"),
    ("A question that needs two policies: travel (hotel limit) and expenses (who approves).",
     ASK % "I stayed in a hotel costing NGN 70,000 per night on a domestic work trip. Is that within the limit, and who needs to approve my claim?"),
    ("Applying a rule to a situation: the gift is above the NGN 15,000 limit, so it must be declared.",
     ASK % "A supplier sent me a gift worth NGN 25,000. Can I keep it?"),
    ("Two policies again: training repayment (learning policy) and leave payout (annual leave policy).",
     ASK % "The company paid NGN 300,000 for my certification. If I resign 8 months after finishing it, what do I owe, and what happens to my unused annual leave?"),
    ("Not covered by any policy: the assistant refuses instead of guessing, with no citations.",
     ASK % "Can I bring my dog to the office?"),
    ("Invalid input (empty question) returns HTTP 422 with an error body, not a fake refusal.",
     'curl -s -w "\\nHTTP %{http_code}\\n" -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d \'{"question": ""}\''),
]


@dataclass
class Frame:
    lines: list[tuple[str, tuple]] = field(default_factory=list)  # (text, colour)
    caption: str = ""
    title: list[str] | None = None
    offset: int | None = None  # first visible line; None = follow the bottom like a terminal


def wrap(text: str, colour) -> list[tuple[str, tuple]]:
    out = []
    for raw in text.splitlines() or [""]:
        indent = len(raw) - len(raw.lstrip(" "))
        parts = textwrap.wrap(raw, COLS, subsequent_indent=" " * (indent + 4), break_long_words=True,
                              break_on_hyphens=False) or [""]
        out += [(p, colour) for p in parts]
    return out


def render(f: Frame) -> bytes:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if f.title:
        y = H // 2 - 40 * len(f.title)
        for i, t in enumerate(f.title):
            font = TITLE if i == 0 else CAP
            tw = d.textlength(t, font=font)
            d.text(((W - tw) / 2, y), t, font=font, fill=FG if i == 0 else DIM)
            y += 70 if i == 0 else 40
    else:
        d.rectangle([0, 0, W, 34], fill=(40, 43, 54))
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([16 + i * 22, 11, 28 + i * 22, 23], fill=c)
        d.text((W / 2 - 110, 8), "Terminal - policy-qa-assistant", font=MONO, fill=DIM)
        visible = f.lines[-MAX_LINES:] if f.offset is None else f.lines[f.offset:f.offset + MAX_LINES]
        for i, (t, colour) in enumerate(visible):
            d.text((LEFT, TOP + i * LINE_H), t, font=MONO, fill=colour)
    if f.caption:
        lines = textwrap.wrap(f.caption, 80)
        bh = 22 + 34 * len(lines)
        d.rectangle([0, H - bh - 12, W, H], fill=CAP_BG)
        for i, t in enumerate(lines):
            tw = d.textlength(t, font=CAP)
            d.text(((W - tw) / 2, H - bh + 2 + i * 34), t, font=CAP, fill=(255, 255, 255))
    return img.tobytes()


class Video:
    def __init__(self, path: Path):
        self.t = 0.0
        self.cues: list[tuple[float, float, str]] = []
        self._cue_start, self._cue_text = 0.0, ""
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(path)],
            stdin=subprocess.PIPE)

    def hold(self, f: Frame, seconds: float) -> None:
        if f.caption != self._cue_text:
            if self._cue_text:
                self.cues.append((self._cue_start, self.t, self._cue_text))
            self._cue_start, self._cue_text = self.t, f.caption
        data = render(f)
        for _ in range(max(1, round(seconds * FPS))):
            self.proc.stdin.write(data)
        self.t += max(1, round(seconds * FPS)) / FPS

    def type_command(self, f: Frame, cmd: str) -> None:
        base = list(f.lines)
        step = max(1, len(cmd) // 45)  # ~3 s of typing for long commands
        for i in range(0, len(cmd) + 1, step):
            f.lines = base + wrap(PROMPT + cmd[:i], FG)
            self.hold(f, 1 / FPS)
        f.lines = base + wrap(PROMPT + cmd, FG)
        self.hold(f, 0.5)

    def close(self) -> None:
        if self._cue_text:
            self.cues.append((self._cue_start, self.t, self._cue_text))
        self.proc.stdin.close()
        self.proc.wait()


def srt_time(s: float) -> str:
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def read_seconds(text: str) -> float:
    """Time to hold an output on screen so it can be read."""
    return min(22.0, max(7.0, len(text.split()) / 3.0))


def main() -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}
    server = subprocess.Popen([sys.executable, "run.py"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    log: list[str] = []
    try:
        for _ in range(120):
            line = server.stdout.readline()
            if line and "symlink" not in line and "warn" not in line.lower():
                log.append(line.rstrip().replace(str(ROOT), "."))
            if "Uvicorn running on" in line:
                break
        httpx.get("http://localhost:8000/health", timeout=10).raise_for_status()

        results = []
        for caption, cmd in SCENES:
            t0 = time.perf_counter()
            out = subprocess.run([BASH, "-c", cmd], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
            results.append((caption, cmd, out.rstrip(), time.perf_counter() - t0))
    finally:
        server.terminate()

    v = Video(OUT / "demo.mp4")
    v.hold(Frame(title=["Company Policy Q&A Assistant", "RAG with verified citations and sentence-level grounding - FastAPI + Claude Opus 5.5",
                        "Live recording: every response below is real service output"],
                 caption="Demo: start the service, then ask questions from the terminal."), 5)

    f = Frame(caption="Start the service with one command. It listens on port 8000 (set PORT to change it).")
    v.hold(f, 1)
    v.type_command(f, "python run.py")
    for line in log:
        f.lines += wrap(line, DIM)
        v.hold(f, 0.35)
    f.caption = "Startup: 10 policies split into 48 sections and indexed (embeddings + BM25); grounding check on."
    v.hold(f, 5)

    for caption, cmd, out, secs in results:
        f = Frame(caption=caption)
        v.hold(f, 1)
        v.type_command(f, cmd)
        f.lines += wrap(out, GREEN if '"answered": true' in out else BLUE)
        overflow = len(f.lines) - MAX_LINES
        if overflow <= 0:
            v.hold(f, read_seconds(out) + 1.5)
        else:
            # Too long for one screen: show the top (command, question, answer) first,
            # then scroll down slowly to the citations, like a reader would.
            f.offset = 0
            v.hold(f, read_seconds(out) * 0.6)
            for off in range(1, overflow + 1):
                f.offset = off
                v.hold(f, 0.15)
            v.hold(f, read_seconds(out) * 0.4 + 1.5)

    v.hold(Frame(title=["Results on the 15 test questions",
                        "Final version: 12 correct answers  |  3 correct refusals  |  0 failures",
                        "Earlier versions: v1 had 1 partially correct (Q2), v2a had 1 incorrect (Q5)",
                        "Details: README.md, results.csv, WRITEUP.md"],
                 caption="All three versions and their failures are reported in README.md and WRITEUP.md."), 6)
    v.close()

    (OUT / "demo.srt").write_text(
        "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n" for i, (a, b, t) in enumerate(v.cues, 1)),
        encoding="utf-8")
    print(f"Wrote {OUT / 'demo.mp4'} ({v.t:.0f} s) and {OUT / 'demo.srt'}")
    for caption, _, _, secs in results:
        print(f"  {secs:4.1f}s  {caption[:70]}")


if __name__ == "__main__":
    main()
