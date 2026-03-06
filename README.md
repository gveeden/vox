# CleverNote - Voice-to-Text with One Keystroke

A fast, lightweight voice transcription tool supporting multiple state-of-the-art ASR models. Press a hotkey to record, and your speech is instantly transcribed and pasted where you need it.

## Features

- 🎙️ **One-key recording**: Hold Ctrl+Space to record, release to transcribe (push-to-talk)
- 🤖 **Multiple ASR backends**: Parakeet, Moonshine, SenseVoice support
- 🌍 **Multilingual**: Support for multiple languages depending on model choice
- 📋 **Smart auto-paste**: Automatically detects terminals for correct paste method (Ctrl+Shift+V vs Ctrl+V)
- 🧠 **Optional LLM post-processing**: Clean up transcriptions with a local ONNX model or cloud API (Anthropic, OpenAI, Gemini)
- 📜 **History tracking**: Last 100 transcripts saved to `history.json` (no audio files)
- ⚡ **Fast daemon mode**: Pre-loaded model for instant transcription
- ⌨️ **Near-zero paste latency**: Persistent uinput device eliminates per-paste Wayland registration delay
- 🔒 **Privacy first**: Runs 100% locally by default; cloud LLM is opt-in
- 🎯 **Wayland native**: Tested on Hyprland; works on Sway and other compositors
- 📦 **On-demand model downloads**: Models downloaded from HuggingFace on first use

## Requirements

- Rust 1.70 or later
- Microphone access

### Linux System Dependencies

**Fedora / RHEL / CentOS:**
```bash
sudo dnf install -y \
  gtk3-devel \
  gdk-pixbuf2-devel \
  cairo-gobject-devel \
  glib2-devel \
  pkgconf-pkg-config
```

**Ubuntu / Debian:**
```bash
sudo apt install -y \
  libgtk-3-dev \
  libgdk-pixbuf-2.0-dev \
  libcairo-gobject2 \
  libglib2.0-dev \
  pkg-config
```

## Installation

### Linux (Quick Install)

```bash
git clone https://github.com/yourusername/clevernote
cd clevernote
chmod +x install_linux.sh
./install_linux.sh
```

The installer will:
- Build the daemon and client binaries
- Install to `~/.local/bin/`
- Optionally set up systemd service
- Configure compositor-specific hotkeys (Hyprland/Sway)

**Enable auto-paste** (required for Ctrl+V injection):
```bash
sudo setcap "cap_dac_override+p" $(which clevernote-daemon)
```

### Manual Installation

```bash
cargo build --release --bin clevernote-daemon --bin clevernote
cp target/release/clevernote-daemon ~/.local/bin/
cp target/release/clevernote ~/.local/bin/
```

## Usage

### Push-to-talk (recommended)

Bind two keys in your compositor — one to start, one to stop:

**Hyprland (`~/.config/hypr/hyprland.conf`):**
```
# Basic: transcribe only
bind = CTRL, SPACE, exec, clevernote start
bindr = CTRL, SPACE, exec, clevernote stop

# With LLM cleanup (uses configured LLM)
bind = CTRL SHIFT, SPACE, exec, clevernote start --llm
bindr = CTRL SHIFT, SPACE, exec, clevernote stop

# With a custom one-off prompt
bind = SUPER, SPACE, exec, clevernote start --prompt "Format as bullet points: {text}"
bindr = SUPER, SPACE, exec, clevernote stop
```

### Toggle mode

```bash
clevernote toggle   # start if stopped, stop if recording
```

### CLI

```bash
clevernote status   # check daemon status
clevernote quit     # stop daemon
clevernote model list           # list available models
clevernote model pull <id>      # download a model
clevernote model set <id>       # switch active ASR model
```

## Configuration

On first run the daemon writes a template to `~/.config/clevernote/config.toml` with all options commented out. Edit it to enable what you need:

```toml
# Transcription model to use.
active_model = "parakeet-tdt-0.6b-v3-int8"

# Automatically paste after transcription (default: true).
auto_paste = true

# Number of CPU threads for inference (default: 4).
# intra_threads = 4

# Optional: microphone override.
# input_device = "Built-in Microphone"

# ── LLM post-processing ────────────────────────────────────────────────────
# Run LLM on every recording by default. Can also be triggered per-recording
# with `clevernote start --llm` regardless of this setting.
# process_transcription = false

# Prompt template. {text} is replaced with the raw transcript.
# process_prompt = "Clean up the following voice transcription: ..."

# ── Option A: Local ONNX model (private, offline) ──────────────────────────
# llm_model = "qwen3.5-0.8b-fp16"

# ── Option B: Cloud API ────────────────────────────────────────────────────

# Anthropic Claude
# llm_api_provider = "anthropic"
# llm_api_key = "sk-ant-..."
# llm_api_model = "claude-haiku-4-5"

# OpenAI
# llm_api_provider = "openai"
# llm_api_key = "sk-..."
# llm_api_model = "gpt-4o-mini"

# Google Gemini
# llm_api_provider = "gemini"
# llm_api_key = "AIza..."
# llm_api_model = "gemini-2.0-flash"
```

## ASR Models

Models are stored in `~/.config/clevernote/models/` and downloaded on demand.

### Available Models

**Parakeet (NVIDIA)** — High accuracy English ASR (default)
| ID | Size | Notes |
|---|---|---|
| `parakeet-tdt-0.6b-v3-int8` | 900 MB | Default, best accuracy/speed |
| `parakeet-tdt-0.6b-v3` | 2.4 GB | FP32 |

**Moonshine (UsefulSensors)** — Ultra-fast streaming ASR
| ID | Size | Notes |
|---|---|---|
| `moonshine-tiny` | 110 MB | ~25× faster than real-time |
| `moonshine-base` | 245 MB | Better accuracy |

**SenseVoice (Alibaba)** — Multilingual with emotion detection
| ID | Size | Notes |
|---|---|---|
| `sensevoice-small` | 937 MB | Chinese / Japanese / Korean / English |

```bash
clevernote model list           # see all models + download status
clevernote model pull <id>      # download
clevernote model set <id>       # switch (restarts daemon)
```

## LLM Post-processing

CleverNote can optionally clean up transcriptions through an LLM. The LLM worker runs in the background and is triggered either per-recording (`--llm` flag) or automatically for every recording (`process_transcription = true`).

**Priority:** local ONNX model › cloud API. If only one is configured, that one is used.

### Option A — Local ONNX model (private)

No API key, no network access, runs on CPU.

```bash
# Download a model first
clevernote model pull qwen3.5-0.8b-fp16

# Then set in config.toml:
# llm_model = "qwen3.5-0.8b-fp16"
```

Available LLM models:

| ID | Size | Notes |
|---|---|---|
| `qwen3.5-0.8b-fp16` | ~2 GB | Best quality local option |
| `qwen3.5-0.8b-q4` | ~670 MB | Quantized, faster |
| `qwen3.5-2b-q4` | ~1.5 GB | Larger, more capable |
| `qwen3.5-4b-q4` | ~2.5 GB | Largest local option |
| `gemma-3-270m-it-q4` | ~800 MB | Google Gemma 3 270M |
| `gemma-3-270m-it-fp16` | ~570 MB | Google Gemma 3 270M FP16 |
| `gemma-3-1b-it-q4` | ~1.7 GB | Google Gemma 3 1B |

### Option B — Cloud API

Set `llm_api_key` and optionally `llm_api_provider` and `llm_api_model`.

| Provider | `llm_api_provider` | Default model | Key format |
|---|---|---|---|
| Anthropic Claude | `anthropic` (default) | `claude-haiku-4-5` | `sk-ant-...` |
| OpenAI | `openai` | `gpt-4o-mini` | `sk-...` |
| Google Gemini | `gemini` | `gemini-2.0-flash` | `AIza...` |

### Per-recording LLM override

```bash
# Use LLM for this recording (regardless of process_transcription setting)
clevernote start --llm

# Use a custom prompt for this recording
clevernote start --prompt "Summarise in one sentence: {text}"
```

## Architecture

- **Audio**: `cpal` for cross-platform audio capture
- **ASR inference**: ONNX Runtime (`ort`)
- **Keyboard injection**: `evdev` uinput (Linux) / `enigo` (macOS/Windows)
- **Clipboard**: `wl-copy` (Wayland) / `arboard` (X11, macOS, Windows)
- **LLM inference**: local ONNX backends (Qwen3.5, Gemma 3) or cloud REST APIs
- **IPC**: newline-delimited JSON over a Unix domain socket
- **CLI**: `clap`

## Troubleshooting

### Auto-paste not working
```bash
# Grant uinput access
sudo setcap "cap_dac_override+p" $(which clevernote-daemon)
```

### Model download fails
```bash
clevernote model pull <id>   # retry download
```

### LLM not running
- Local model: check it is downloaded (`clevernote model list`) and `llm_model` is set in config
- API: check the API key is correct and `llm_api_provider` matches the key type

### No input device found
- Check microphone permissions in system settings
- Use `clevernote device list` to see available devices, then set `input_device` in config

## License

MIT / Apache-2.0

## Credits

- [cpal](https://github.com/RustAudio/cpal) — Cross-platform audio capture
- [ort](https://github.com/pykeio/ort) — ONNX Runtime Rust bindings
- [evdev](https://github.com/emberian/evdev) — Linux input device access
