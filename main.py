#!/usr/bin/env python3
"""
yt_subber.py
-------------
Given a file path OR a video URL:
- If URL: download with yt-dlp.
- Then: check for English subtitles:
   * Use sibling "<video>.en.srt" if present, else
   * Extract embedded English subs with ffmpeg/ffprobe if present, else
   * Use OpenAI Whisper API to generate SRT (translate by default; --transcribe to keep source language)
- Audio is extracted as MP3 (mono, 16kHz, 64 kbps CBR) then split into <25MB chunks.
- Chunks are sent sequentially to Whisper (`whisper-1`) with response_format="srt", optional --prompt.
- Resulting SRTs are stitched with proper timestamp offsets and saved as "<video>.en.srt".
- Basic SRT validation; on failure, the file is deleted (no retry).

Requirements (CLI tools): ffmpeg, ffprobe, yt-dlp (for URLs).
Python: tqdm, requests (if openai package not available), but we use openai>=1.0 preferred.
Environment: export OPENAI_API_KEY=...

Usage:
  python yt_subber.py "<path-or-video-url>" [--transcribe] [--prompt "style or terms"] [--workdir /tmp]
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yt_dlp
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("yt_subber")


def run(
    cmd: List[str], check=True, capture_output=False, text=True, env=None
) -> subprocess.CompletedProcess:
    """Run a subprocess with nice error messages."""
    try:
        return subprocess.run(
            cmd, check=check, capture_output=capture_output, text=text, env=env
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        stdout = e.stdout or ""
        raise RuntimeError(
            f"Command failed: {' '.join(shlex.quote(c) for c in cmd)}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        ) from e


def is_url(s: str) -> bool:
    return bool(re.match(r"^(https?://)?([a-z0-9-]+\.)+[a-z]{2,}(/.*)?$", s, re.I))


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_")


def ffprobe_streams(video_path: Path) -> List[dict]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,codec_name:stream_tags=language,title",
        "-of",
        "json",
        str(video_path),
    ]
    out = run(cmd, capture_output=True).stdout
    data = json.loads(out)
    return data.get("streams", [])


def extract_embedded_english_subs(video_path: Path, out_srt: Path) -> bool:
    streams = ffprobe_streams(video_path)
    subs = []
    for s in streams:
        if s.get("codec_type") != "subtitle":
            continue
        lang = (s.get("tags", {}) or {}).get("language", "") or ""
        if lang.lower() in {"eng", "en"}:
            subs.append(s["index"])

    if not subs:
        return False

    # Map first English subtitle stream; transcode to SRT if needed
    stream_index = subs[0]
    tmp_srt = out_srt.with_suffix(".tmp.en.srt")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-map",
        f"0:{stream_index}",
        "-c:s",
        "srt",
        str(tmp_srt),
    ]
    run(cmd, check=True, capture_output=True)
    if tmp_srt.exists() and tmp_srt.stat().st_size > 0:
        tmp_srt.replace(out_srt)
        return True
    return False


BITRATE_BPS = 64_000  # 64 kbps CBR
SPLIT_TARGET_MB = 24.0  # keep a margin below 25 MB


def extract_audio_mp3(video_path: Path, audio_path: Path) -> None:
    # mono, 16 kHz, 64 kbps CBR -> very speech-friendly, predictable size
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        f"{BITRATE_BPS // 1000}k",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        str(audio_path),
    ]
    run(cmd, check=True, capture_output=True)


def compute_chunk_seconds() -> int:
    # Bytes per second = BITRATE_BPS / 8. Use SPLIT_TARGET_MB MB cap.
    bytes_per_sec = BITRATE_BPS / 8.0
    max_bytes = SPLIT_TARGET_MB * 1024 * 1024
    seconds = int(max_bytes // bytes_per_sec) - 5  # small safety margin
    return max(seconds, 60)  # at least 1 minute


def split_audio_to_chunks(audio_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_seconds = compute_chunk_seconds()
    # Use segment muxer to split by time
    out_pattern = out_dir / "chunk_%04d.mp3"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        str(out_pattern),
    ]
    run(cmd, check=True, capture_output=True)
    chunks = sorted(out_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("Audio splitting produced no chunks.")
    return chunks


@dataclass
class SRTEntry:
    idx: int
    start_ms: int
    end_ms: int
    text: str


TIME_RE = re.compile(r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})")


def parse_timecode(s: str) -> int:
    m = TIME_RE.fullmatch(s.strip())
    if not m:
        raise ValueError(f"Bad timecode: {s}")
    h = int(m.group("h"))
    m_ = int(m.group("m"))
    s_ = int(m.group("s"))
    ms = int(m.group("ms"))
    return ((h * 60 + m_) * 60 + s_) * 1000 + ms


def fmt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(text: str) -> List[SRTEntry]:
    entries = []
    blocks = re.split(r"\r?\n\r?\n", text.strip(), flags=re.MULTILINE)
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0].strip())
        except ValueError:
            # Sometimes index is missing; try to recover
            idx = (entries[-1].idx + 1) if entries else 1
            timing = lines[0]
            text_lines = lines[1:]
        else:
            timing = lines[1] if len(lines) > 1 else ""
            text_lines = lines[2:]
        if "-->" not in timing:
            # Try next line
            if len(lines) >= 2 and "-->" in lines[1]:
                timing = lines[1]
                text_lines = lines[2:]
            else:
                raise ValueError(f"Bad SRT timing line: {timing}")
        start_str, end_str = [t.strip() for t in timing.split("-->")]
        start_ms = parse_timecode(start_str)
        end_ms = parse_timecode(end_str.split()[0])
        text_block = "\n".join(text_lines).strip()
        entries.append(SRTEntry(idx, start_ms, end_ms, text_block))
    return entries


def serialize_srt(entries: List[SRTEntry]) -> str:
    lines = []
    for i, e in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_time(e.start_ms)} --> {fmt_time(e.end_ms)}")
        lines.append(e.text)
        lines.append("")  # blank line
    return "\n".join(lines).strip() + "\n"


def validate_srt(entries: List[SRTEntry]) -> Tuple[bool, str]:
    if not entries:
        return False, "Empty SRT"
    prev_end = -1
    for i, e in enumerate(entries, start=1):
        if e.end_ms < e.start_ms:
            return False, f"Entry {i} has end before start"
        if e.start_ms < prev_end - 200:  # allow small overlaps due to rounding
            return False, f"Entry {i} starts before previous end"
        prev_end = max(prev_end, e.end_ms)
        if not e.text.strip():
            return False, f"Entry {i} has empty text"
    return True, "OK"


def offset_entries(entries: List[SRTEntry], offset_ms: int) -> List[SRTEntry]:
    out = []
    for e in entries:
        out.append(
            SRTEntry(e.idx, e.start_ms + offset_ms, e.end_ms + offset_ms, e.text)
        )
    return out


def openai_client():
    return OpenAI()


def whisper_request(
    file_path: Path, translate: bool, prompt: Optional[str], retries: int
) -> str:
    """
    Returns the SRT text for the given audio chunk.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    client = openai_client()
    attempt = 0
    while True:
        try:
            with file_path.open("rb") as f:
                if translate:
                    # English translation from source language
                    resp = client.audio.translations.create(
                        model="whisper-1",
                        file=f,
                        response_format="srt",
                        prompt=prompt or None,
                    )
                else:
                    # Transcription in source language
                    resp = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="srt",
                        prompt=prompt or None,
                    )
                return (
                    resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
                )
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise e
            import random
            import time as _t

            _t.sleep(min(2**attempt, 10) + random.random())


def download_video(url: str, workdir: Path) -> Path:
    """
    Download the video using yt-dlp's Python API. Returns the downloaded file path.
    """
    # Download best mp4/mkv as single file, write to workdir
    workdir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(workdir / "%(title)s [%(id)s].%(ext)s")

    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Pull English subs (official or auto) alongside the video:
        "writesubtitles": True,
        "writeautomaticsub": False,
        # accept en and region variants (en-US, en-GB, …)
        "subtitleslangs": ["en", "en.*"],
        # request SRT; if source is VTT/TTML, convert to SRT
        "subtitlesformat": "srt",
        "postprocessors": [
            {"key": "FFmpegSubtitlesConvertor", "format": "srt"},
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_path = Path(info.get("filepath") or ydl.prepare_filename(info)).resolve()
    if not video_path.exists():
        raise RuntimeError(
            "yt-dlp reported no output file (filepath/prepare_filename not found on disk)."
        )
    return video_path


def ensure_video(input_arg: str, workdir: Path) -> Path:
    """
    If URL: download via yt-dlp library; else resolve local file.
    """
    if is_url(input_arg):
        log.info("URL provided, starting download...")
        return download_video(input_arg, workdir)
    p = Path(input_arg).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    return p


def existing_sidecar_srt(video_path: Path) -> Optional[Path]:
    exact = video_path.with_suffix(".en.srt")
    if exact.exists() and exact.stat().st_size > 0:
        return exact
    # accept en-XX / en_XX sidecars and normalize to .en.srt
    stem = video_path.with_suffix("").name
    parent = video_path.parent
    matches = sorted(
        parent.glob(f"{stem}.en*.srt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if matches:
        tmp = matches[0]
        tmp.replace(exact) if not exact.exists() else None
        return exact if exact.exists() else tmp
    return None


def transcribe_via_whisper(
    video_path: Path,
    out_srt: Path,
    translate_default: bool,
    user_prompt: Optional[str],
    retries: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="ytw_") as tmpd:
        tmpdir = Path(tmpd)
        audio_path = tmpdir / "audio.mp3"
        extract_audio_mp3(video_path, audio_path)

        # Split if needed
        chunks = split_audio_to_chunks(audio_path, tmpdir / "chunks")

        combined_entries: List[SRTEntry] = []
        offset_ms_total = 0

        with tqdm(total=len(chunks), desc="Transcribing", unit="chunk") as pbar:
            for idx, ch in enumerate(chunks, start=1):
                srt_text = whisper_request(
                    ch,
                    translate=translate_default,
                    prompt=user_prompt,
                    retries=retries,
                )
                # Parse -> offset -> append
                entries = parse_srt(srt_text)  # let it raise; top-level catches
                # Estimate chunk duration for offset: get from ffprobe (accurate) to avoid CBR assumptions
                dur_ms = probe_duration_ms(ch)
                shifted = offset_entries(entries, offset_ms_total)
                combined_entries.extend(shifted)
                offset_ms_total += dur_ms
                pbar.update(1)
                # Small nap to be nice to rate limits on some accounts
                time.sleep(0.2)

        # Validate & write
        ok, msg = validate_srt(combined_entries)
        if not ok:
            # Write a debug file alongside to help diagnose (but not final .en.srt)

            log.error(f"SRT validation failed: {msg}. Still wrote subtitles file.")
        out_srt.write_text(serialize_srt(combined_entries), encoding="utf-8")


def probe_duration_ms(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = run(cmd, capture_output=True).stdout.strip()
    try:
        seconds = float(out)
    except Exception as e:
        raise RuntimeError(f"ffprobe failed to parse duration for {path}: {out}") from e
    return int(seconds * 1000)


def main():
    parser = argparse.ArgumentParser(
        description="Download (optional), detect/extract subtitles, otherwise generate with Whisper API."
    )
    parser.add_argument("input", help="Path to local video file OR video URL")
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="Transcribe in source language (default is to translate to English)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional prompt to guide Whisper (e.g., names, spellings, style)",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=None,
        help="Working directory (for yt-dlp downloads). Defaults to the video's folder.",
    )
    parser.add_argument("--retries", type=int, default=3, help="API retry attempts")
    args = parser.parse_args()

    # Basic logging to STDERR with timestamps

    try:
        workdir = Path(args.workdir).expanduser().resolve() if args.workdir else None
        if workdir:
            workdir.mkdir(parents=True, exist_ok=True)
        video_path = ensure_video(args.input, workdir or Path.cwd())
        if not workdir:
            workdir = video_path.parent

        log.info(f"Video: {video_path}")

        # Step 1: If sidecar exists, done.
        sidecar = existing_sidecar_srt(video_path)
        if sidecar:
            log.info(f"Found existing sidecar: {sidecar.name}. Nothing to do.")
            return

        # Step 2: Try extracting embedded English subs.
        out_srt = video_path.with_suffix(".en.srt")
        log.info("Checking for embedded English subtitles...")
        if extract_embedded_english_subs(video_path, out_srt):
            log.info(f"Extracted embedded English subtitles -> {out_srt.name}")
            return

        # Step 3: Generate with Whisper.
        mode = "transcribe" if args.transcribe else "translate"
        log.info(f"No subtitles found. Calling Whisper API to {mode}...")
        transcribe_via_whisper(
            video_path,
            out_srt,
            translate_default=(not args.transcribe),
            user_prompt=args.prompt,
            retries=args.retries,
        )
        log.info(f"Wrote {out_srt.name}")

    except KeyboardInterrupt:
        print("Aborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        log.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
