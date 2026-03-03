# CleverNote - Voice-to-Text with One Keystroke

A fast, lightweight voice transcription tool supporting multiple state-of-the-art ASR models. Press a hotkey to record, and your speech is instantly transcribed and pasted where you need it.

## Features

- 🎙️ **One-key recording**: Press Ctrl+Space to toggle recording (configurable)
- 🤖 **Multiple ASR backends**: Moonshine and SenseVoice support
- 🌍 **Multilingual**: Support for multiple languages depending on model choice
- 📋 **Smart auto-paste**: Automatically detects terminals and apps for correct paste method
- 📜 **History tracking**: Last 100 transcripts saved to `history.json` (no audio files)
- 🔔 **Clean notifications**: Single notification while recording (Linux: notify-send)
- ⚡ **Fast daemon mode**: Pre-loaded model for instant transcription
- 🔒 **Privacy**: Runs 100% locally, no cloud services
- 🖥️ **Cross-platform**: macOS (untested), Linux (X11/Wayland)
- 🎯 **Wayland native**: Works perfectly on Hyprland (tested), Sway (untested), and other Wayland compositors (untested)
- 📦 **On-demand model downloads**: Models downloaded automatically from HuggingFace
- Limited to models in list

## Requirements

- Rust 1.70 or later
- ONNX model files (see Model Setup below)
- Microphone access

### Linux System Dependencies

Install the required GTK/GLib development libraries before building:

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

**Default behavior (recommended)**: 
- Text copied to clipboard via wl-copy (Wayland) or arboard (X11)
- Auto-pasted using evdev virtual keyboard simulation
- Smart terminal detection: Ctrl+Shift+V for terminals, Ctrl+V for other apps
- Requires CAP_DAC_OVERRIDE capability: `sudo setcap cap_dac_override+ep ~/.local/bin/clevernote-daemon`

**Note**: The `auto_inject` config option has been removed. All pasting now uses the evdev method for reliability.

### macOS

```bash
./install_launchagent.sh
```

**Important:** Grant Accessibility permissions for hotkeys and pasting. See [LAUNCHAGENT_SETUP.md](LAUNCHAGENT_SETUP.md).

### Manual Installation

```bash
# Build daemon and client
cargo build --release --features daemon --bin clevernote-daemon --bin clevernote

# Install binaries
cp target/release/clevernote-daemon ~/.local/bin/
cp target/release/clevernote ~/.local/bin/
```

## Model System

CleverNote supports multiple speech recognition backends with automatic model management. Models are downloaded on-demand from HuggingFace and stored in `~/.config/clevernote/models/`.

### Available Models

**Parakeet (NVIDIA)** - High accuracy English ASR (default)
- `parakeet-tdt-0.6b-v3-int8` - 0.6B params, 900MB, INT8 quantized (default)
- `parakeet-tdt-0.6b-v3` - 0.6B params, 2.4GB, FP32

**Moonshine (UsefulSensors)** - Ultra-fast streaming ASR
- `moonshine-tiny` - 27M params, 110MB, ~25x faster than real-time
- `moonshine-base` - 61M params, 245MB, better accuracy

**SenseVoice (Alibaba)** - Multilingual with emotion detection
- `sensevoice-small` - 220M params, 937MB, supports Chinese/Japanese/Korean/English

### Model Configuration

Models are configured in two places:
1. **Embedded defaults** - Built into the binary at compile time (`models.json`)
2. **User overrides** - Runtime config at `~/.config/clevernote/models.json`

User models override or extend embedded defaults (matched by ID).

Example custom model in `~/.config/clevernote/models.json`:
```json
{
  "models": [
    {
      "id": "my-custom-moonshine",
      "name": "My Custom Moonshine",
      "description": "Custom Moonshine model",
      "repository": "username/repo-name",
      "backend": "moonshine",
      "quantization": "fp32",
      "model_type": "multi-file",
      "files": [
        {
          "url": "https://huggingface.co/username/repo-name/resolve/main/encoder.onnx",
          "filename": "encoder.onnx"
        },
        {
          "url": "https://huggingface.co/username/repo-name/resolve/main/decoder.onnx",
          "filename": "decoder.onnx"
        },
        {
          "url": "https://huggingface.co/username/repo-name/resolve/main/tokenizer.json",
          "filename": "tokenizer.json"
        }
      ]
    }
  ]
}
```

### Model Selection

Specify a model ID when starting the daemon:
```bash
# Use Parakeet INT8 (default, best accuracy/speed balance)
clevernote-daemon --model parakeet-tdt-0.6b-v3-int8

# Use Moonshine Tiny (fastest, smallest)
clevernote-daemon --model moonshine-tiny

# Use SenseVoice (Asian languages + emotion)
clevernote-daemon --model sensevoice-small
```

If no model is specified, the daemon uses `parakeet-tdt-0.6b-v3-int8` (the default).

### Model Downloads

Models are downloaded automatically on first use:
```bash
# First run downloads model files with progress bar
clevernote-daemon --model moonshine-tiny
# Downloading moonshine-tiny (110 MB)...
# [====================================] 100% encoder.onnx
# [====================================] 100% decoder.onnx
# [====================================] 100% tokenizer.json

# Subsequent runs load instantly
clevernote-daemon --model moonshine-tiny
# Loading model from ~/.config/clevernote/models/moonshine-tiny/
```

### Model Storage

Models are stored in: `~/.config/clevernote/models/{model-id}/`

Example structure:
```
~/.config/clevernote/
├── models/
│   ├── moonshine-tiny/
│   │   ├── encoder.onnx
│   │   ├── decoder.onnx
│   │   └── tokenizer.json
│   ├── moonshine-base/
│   │   ├── encoder.onnx
│   │   ├── decoder.onnx
│   │   └── tokenizer.json
│   └── sensevoice-small/
│       ├── model.onnx
│       ├── tokenizer.model
│       ├── am.mvn
│       ├── embedding.npy
│       └── vocab.json (generated at runtime)
├── models.json (user overrides)
└── config.toml
```

## Usage

### Quick Start

The easiest way to get started:

```bash
# Start daemon with Moonshine Tiny (fastest, auto-downloads)
clevernote-daemon --model moonshine-tiny

# Or use Moonshine Base (better accuracy)
clevernote-daemon --model moonshine-base

# Or use SenseVoice (Asian languages)
clevernote-daemon --model sensevoice-small
```

Models are downloaded automatically on first use.

### Command Line Options

```
Daemon:
  -m, --model <MODEL_ID>           Model ID from registry (e.g., moonshine-tiny, sensevoice-small)
  -c, --config <FILE>              Path to config file [default: ~/.config/clevernote/config.toml]
  --device <NAME>                  Input device name (runtime override, does not modify config)
  -h, --help                       Print help
  -V, --version                    Print version

Client:
  toggle                           Start/stop recording
  device list                      List local input audio devices (no daemon IPC)
  status                           Check daemon status
  quit                             Stop daemon
  -h, --help                       Print help
```

## Configuration

Configuration file: `~/.config/clevernote/config.toml`

```toml
# Modifier key: Alt, Ctrl, Cmd, or Shift
modifier_key = "Alt"

# Trigger key: Space, Return, Tab, Escape
trigger_key = "Space"

# Whether to automatically paste after transcription (default: true)
# Set to false to only copy to clipboard; paste manually with Ctrl+V
auto_paste = true

# Whether to type text character-by-character via evdev instead of simulating Ctrl+V (default: false)
# Requires: sudo setcap cap_dac_override+ep ~/.local/bin/clevernote-daemon
auto_inject = false

# Optional input device name for daemon recording
# If unset, default_input_device() is used
input_device = "Built-in Microphone"
```

The daemon will create a default config on first run if one doesn't exist.

### Linux-Specific Setup

After installation, grant the daemon permission to simulate keyboard events:

```bash
sudo setcap cap_dac_override+ep ~/.local/bin/clevernote-daemon
```

This allows the daemon to access `/dev/uinput` for auto-paste functionality without running as root. The capability is only used when creating the virtual keyboard device and is dropped immediately after.

### Transcript History

All transcripts are saved to `~/.config/clevernote/transcripts/history.json`:

```json
{
  "transcripts": [
    {
      "timestamp": "20260217_134333",
      "text": "Your transcribed text here"
    }
  ]
}
```

The last 100 transcripts are kept automatically (most recent first). Audio files are **not** saved - only transcription text is stored.

## How to Use

### Daemon Mode (Recommended)

1. **Start the daemon**:
   ```bash
   clevernote-daemon
   ```
   Or use systemd: `systemctl --user start clevernote-daemon`

2. **Use the hotkey** (Ctrl+Space by default):
   - **First press**: Start recording (🔴 Recording notification appears via notify-send)
   - Speak your message
   - **Second press**: Stop recording and transcribe
   - Text is automatically:
     - Copied to clipboard (wl-copy on Wayland)
     - Pasted at cursor (Ctrl+Shift+V for terminals, Ctrl+V for other apps)
     - Saved to history.json

3. **Control with CLI**:
   ```bash
   clevernote toggle   # Start/stop recording
   clevernote status   # Check daemon status
   clevernote quit     # Stop daemon
   ```

## Architecture

- **Audio Recording**: `cpal` for cross-platform audio capture
- **ASR Models**: ONNX Runtime for model inference
- **Keyboard Hotkeys**: `rdev` for global keyboard listener
- **Clipboard**: `arboard` for clipboard operations
- **Notifications**: `notify-rust` for system notifications
- **CLI**: `clap` for argument parsing

## Troubleshooting

### No input device found
- Check microphone permissions in system settings
- Ensure a microphone is connected

### Model loading failed
- Verify all required model files are present
- Check file paths are correct
- Ensure ONNX Runtime can access the files

### Keyboard shortcuts not working
- Check accessibility permissions (macOS)
- Verify no other app is capturing the same hotkey

### Transcription errors
- Ensure audio is clear and loud enough
- Check model compatibility with audio format
- Try with `--keep-audio` flag to debug audio files

## License

This project is dual-licensed under MIT/Apache-2.0.

## Credits

- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio capture
- [ort](https://github.com/pykeio/ort) - ONNX Runtime Rust bindings
