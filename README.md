# translate - Download videos and utilize Whisper to translate them into English

Create clean English `.srt` subtitles for a local video **or** a YouTube URL.

The script follows a simple decision tree:

1. **If a sidecar file exists** (`<video>.en.srt`) → use it, done.  
2. **Else if embedded English captions exist** → extract with `ffmpeg/ffprobe`.  
3. **Else** → generate subtitles with the **OpenAI Whisper API** (`whisper-1`).  
   - Audio is extracted as MP3 (mono, 16 kHz, 64 kbps CBR), split into `<25 MB` chunks.  
   - Chunks are sent sequentially to Whisper with `response_format="srt"`.  
   - SRT chunks are stitched with correct timestamp offsets and saved to `<video>.en.srt`.

> **Note:** The current implementation performs basic SRT validation and logs an error if it detects problems; it still writes the final `.srt`. (You can change this behavior if you prefer to fail hard.)

---

## Requirements

- **Python** 3.10+  
- **uv** (dependency manager & runner)  
  - Install uv (one-time):
    - macOS/Linux:

      ```bash
      curl -LsSf https://astral.sh/uv/install.sh | sh
      ```

    - Windows: see the uv docs or use WSL.
- **CLI tools**
  - `ffmpeg` and `ffprobe`
  - `yt-dlp` (only required when input is a URL)
- **OpenAI API key**
  - Set in your environment: `export OPENAI_API_KEY=sk-...`

The Python dependencies (e.g., `openai`, `yt_dlp`, `tqdm`) are declared in the project’s `pyproject.toml`. `uv sync` installs them into a managed environment.

---

## Quick start

1. **Install system tools** (examples):

    - macOS (Homebrew):

    ```bash
    brew install ffmpeg
    ```

    - Debian/Ubuntu:

    ```bash
    sudo apt-get update
    sudo apt-get install -y ffmpeg
    ```

2. **Install Python deps with uv**:

    ```bash
    uv sync
    ```

3. **Set your API key**:

    ```bash
    export OPENAI_API_KEY=sk-...   # PowerShell: $env:OPENAI_API_KEY="sk-..."
    ```

4. **Run**:

    - From a **YouTube URL**:

    ```bash
    uv run python main.py "https://youtu.be/VIDEO_ID"
    ```

    - From a **local file**:

    ```bash
    uv run python main.py "/path/to/video.mp4"
    ```

---

## CLI usage

```bash
python main.py "<path-or-video-url>" [--transcribe] [--prompt "style or terms"] [--workdir /tmp] [--retries 3]
```

### Options

- `--transcribe`  
  Don’t translate; transcribe in the source language. (Default is **translate to English**.)
- `--prompt "..."`  
  Optional terms/style hints for Whisper (e.g., names, acronyms, proper nouns).
- `--workdir /path`  
  Working directory for downloads and intermediates (defaults to the video’s folder).
- `--retries N`  
  Whisper API retry attempts (default: `3`).

### Examples

```bash
# Translate to English (default), add domain terms
uv run python main.py "https://www.youtube.com/watch?v=abc123" --prompt "Kubernetes, Istio, Terraform"

# Transcribe in the source language
uv run python main.py "/videos/lecture.mkv" --transcribe

# Custom workdir
uv run python main.py "https://youtu.be/xyz789" --workdir /tmp/ytw
```

---

## How it works (details)

1. **Input resolution**
   - If the argument looks like a URL, the script uses `yt-dlp`’s Python API to download the best single-file video into the working directory.  
   - If it’s a file path, it resolves the local file.

2. **Subtitle discovery**
   - Checks for an existing sidecar matching `<video>.en.srt` (also accepts `en-*` variants and normalizes to `.en.srt`).
   - If none, probes streams with `ffprobe`. If any subtitle stream is English, stops further processing.
   - Otherwise, proceeds to utilize Whisper to create English subtitles.

3. **Whisper fallback**
   - Extracts audio: MP3, mono, 16 kHz, 64 kbps CBR.  
   - Splits into time-based chunks sized to stay under ~25 MB.  
   - Sends each chunk to OpenAI Whisper (`whisper-1`) with `response_format="srt"`:
     - **Translate** (default) or **transcribe** (if `--transcribe`).
     - Optional `--prompt` is forwarded to Whisper.
   - Parses chunk `.srt`, offsets timestamps by cumulative chunk durations, and stitches into a single `.en.srt`.

4. **Validation & output**
   - Performs basic structural checks (non-empty, non-overlapping, sane timing).  
   - Logs any validation errors; writes `<video>.en.srt` next to the source video.

---

## Output

- Final subtitles: `<video>.en.srt` (always English when translating; source language when `--transcribe`).
- Intermediates: stored under the working directory (temporary audio and chunks created inside a temp dir during execution).

---

## Logging, retries & exits

- Logs to STDERR with timestamps (`INFO` level).
- Whisper calls retry with exponential backoff (`--retries`, default 3).
- Exit codes:
  - `0` on success
  - `130` on Ctrl-C
  - non-zero on errors

---

## Troubleshooting

- **`ffmpeg` / `ffprobe` not found**  
  Ensure both are installed and on `PATH`. `ffmpeg -version` should work.
- **`yt-dlp` errors**  
  Update to the latest `yt-dlp` using `uv`. Some sites require fresh extractors.
- **`OPENAI_API_KEY` missing**  
  Set the environment variable before running.
- **Large files / rate limits**  
  The script splits audio to avoid size limits. If you still see rate-limit errors, rerun; the script retries automatically.

---

## Security notes

- The OpenAI API key is read from the environment and not stored in files.  
- The script writes subtitles alongside the video; avoid running it in directories you don’t control.
