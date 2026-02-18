# CleverNote - Voice-to-Text with One Keystroke

A fast, lightweight voice transcription tool supporting multiple state-of-the-art ASR models. Press a hotkey to record, and your speech is instantly transcribed and pasted where you need it.

## Features

- 🎙️ **One-key recording**: Press Alt+Space to toggle recording (configurable)
- 🤖 **Multiple ASR backends**: Whisper, Moonshine, SenseVoice, and Parakeet support
- 🌍 **Multilingual**: Support for 100+ languages depending on model choice
- 📋 **Smart auto-paste**: Automatically detects terminals and apps for correct paste method
- 📜 **History tracking**: Last 100 transcripts saved to `history.json` (no audio files)
- 🔔 **Clean notifications**: Single notification while recording (Linux: notify-send)
- ⚡ **Fast daemon mode**: Pre-loaded model for instant transcription
- 🔒 **Privacy**: Runs 100% locally, no cloud services
- 🖥️ **Cross-platform**: macOS, Linux (X11/Wayland), Windows
- 🎯 **Wayland native**: Works perfectly on Hyprland, Sway, and other Wayland compositors
- 📦 **On-demand model downloads**: Models downloaded automatically from HuggingFace

## Requirements

- Rust 1.70 or later
- ONNX model files (see Model Setup below)
- Microphone access

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

**Whisper (OpenAI)** - Best multilingual accuracy, 100+ languages
- `whisper-tiny` - 39M params, 150MB, fastest Whisper
- `whisper-base` - 74M params, 290MB, balanced speed/accuracy
- `whisper-small` - 244M params, 970MB, good accuracy
- `whisper-medium` - 769M params, 3GB, high accuracy
- `whisper-large-v3-turbo` - 809M params, 1.5GB, best Whisper with timestamps

**Moonshine (UsefulSensors)** - Ultra-fast streaming ASR
- `moonshine-tiny` - 27M params, 110MB, ~25x faster than real-time
- `moonshine-base` - 61M params, 245MB, better accuracy

**SenseVoice (Alibaba)** - Multilingual with emotion detection
- `sensevoice-small` - 220M params, 937MB, supports Chinese/Japanese/Korean/English

**Parakeet (NVIDIA)** - Currently unavailable (models not publicly hosted)

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
      "id": "my-custom-whisper",
      "name": "My Custom Whisper",
      "description": "Fine-tuned Whisper for my use case",
      "repository": "username/repo-name",
      "backend": "whisper",
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
        },
        {
          "url": "https://huggingface.co/username/repo-name/resolve/main/vocab.json",
          "filename": "vocab.json"
        }
      ]
    }
  ]
}
```

### Model Selection

Specify a model ID when starting the daemon:
```bash
# Use Moonshine Tiny (fastest)
clevernote-daemon --model moonshine-tiny

# Use Whisper Small (balanced)
clevernote-daemon --model whisper-small

# Use SenseVoice (Asian languages + emotion)
clevernote-daemon --model sensevoice-small
```

If no model is specified, the daemon uses the model marked as `"default": true` in the registry (currently `parakeet-tdt-0.6b-v3-int8`, though unavailable).

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
│   ├── whisper-small/
│   │   ├── encoder.onnx
│   │   ├── decoder.onnx
│   │   ├── vocab.json
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

# Or use Whisper Small (better accuracy)
clevernote-daemon --model whisper-small

# Or use SenseVoice (Asian languages)
clevernote-daemon --model sensevoice-small
```

Models are downloaded automatically on first use.

### Command Line Options

```
Daemon:
  -m, --model <MODEL_ID>           Model ID from registry (e.g., moonshine-tiny, whisper-small)
  -c, --config <FILE>              Path to config file [default: ~/.config/clevernote/config.toml]
  -h, --help                       Print help
  -V, --version                    Print version

Client:
  toggle                           Start/stop recording
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

2. **Use the hotkey** (Alt+Space by default):
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

### Standalone Mode (No Daemon)

```bash
# Quick transcription (parakeet CLI)
./parakeet --model models/parakeet-tdt
```

Press Alt+Space to toggle recording. Text transcribed and pasted automatically.

## Hardware Acceleration

Enable GPU acceleration by building with feature flags:

```bash
# CUDA
cargo build --release --features cuda

# TensorRT
cargo build --release --features tensorrt

# CoreML (macOS)
cargo build --release --features coreml

# DirectML (Windows)
cargo build --release --features directml

# ROCm (AMD)
cargo build --release --features rocm

# OpenVINO
cargo build --release --features openvino

# WebGPU
cargo build --release --features webgpu
```

## Examples

### Fast Transcription (Moonshine)
```bash
# Fastest model, ~25x faster than real-time
clevernote-daemon --model moonshine-tiny
```

### Accurate Multilingual (Whisper)
```bash
# Best accuracy, 100+ languages
clevernote-daemon --model whisper-small

# Even better accuracy
clevernote-daemon --model whisper-medium
```

### Asian Languages + Emotion (SenseVoice)
```bash
# Chinese, Japanese, Korean, English with emotion detection
clevernote-daemon --model sensevoice-small
```

### Performance Comparison

Tested on typical voice recordings (~10-30 seconds):

| Model | Speed | Accuracy | Languages | Size |
|-------|-------|----------|-----------|------|
| Moonshine Tiny | ⚡⚡⚡⚡⚡ ~25x RT | ⭐⭐⭐ | English | 110 MB |
| Moonshine Base | ⚡⚡⚡⚡ ~20x RT | ⭐⭐⭐⭐ | English | 245 MB |
| Whisper Tiny | ⚡⚡⚡ ~10x RT | ⭐⭐⭐ | 100+ | 150 MB |
| Whisper Small | ⚡⚡ ~5x RT | ⭐⭐⭐⭐ | 100+ | 970 MB |
| Whisper Medium | ⚡ ~2x RT | ⭐⭐⭐⭐⭐ | 100+ | 3 GB |
| SenseVoice | ⚡⚡ ~2.6x RT | ⭐⭐⭐⭐ | 5 Asian | 937 MB |

*RT = Real-time (1x = same duration as audio)*

## Differences from Python Version

This Rust implementation offers several advantages:

- **Flexible Model Support**: Specify any ONNX model path via command line
- **Better Performance**: Native Rust with optimized ONNX Runtime
- **Lower Memory Usage**: Efficient memory management
- **Cross-platform**: Works on macOS, Linux, and Windows
- **No Python Dependency**: Single binary, no runtime required

## Architecture

- **Audio Recording**: `cpal` for cross-platform audio capture
- **ASR Models**: `parakeet-rs` library for ONNX inference
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
- Try running with administrator privileges (Windows)
- Verify no other app is capturing the same hotkey

### Transcription errors
- Ensure audio is clear and loud enough
- Check model compatibility with audio format
- Try with `--keep-audio` flag to debug audio files

## License

This project uses the parakeet-rs library which is dual-licensed under MIT/Apache-2.0.

## Credits

- [parakeet-rs](https://github.com/altunenes/parakeet-rs) - Rust bindings for Parakeet
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Original Parakeet models
