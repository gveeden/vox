# CleverNote - Voice-to-Text with One Keystroke

A fast, lightweight voice transcription tool using NVIDIA Parakeet ASR models. Press a hotkey to record, and your speech is instantly transcribed and pasted where you need it.

## Features

- 🎙️ **One-key recording**: Press Alt+Space to toggle recording (configurable)
- 🤖 **Multilingual**: Supports 25+ languages via Parakeet TDT model
- 📋 **Smart auto-paste**: Automatically detects terminals and apps for correct paste method
- 📜 **History tracking**: Last 100 transcripts saved to `history.json` (no audio files)
- 🔔 **Clean notifications**: Single notification while recording (Linux: notify-send)
- ⚡ **Fast daemon mode**: Pre-loaded model for instant transcription
- 🔒 **Privacy**: Runs 100% locally, no cloud services
- 🖥️ **Cross-platform**: macOS, Linux (X11/Wayland), Windows
- 🎯 **Wayland native**: Works perfectly on Hyprland, Sway, and other Wayland compositors

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

## Model Setup

### Option 1: CTC Model (English-only)

Download the ONNX model files to a directory. Required files:
- `model.onnx` (or `model_fp16.onnx`, `model_int8.onnx`, `model_q4.onnx`)
- `tokenizer.json`
- `config.json`
- `preprocessor_config.json`

### Option 2: TDT Model (Multilingual)

For the TDT model (supports 25+ languages), you need:
- `encoder-model.onnx`
- `decoder_joint-model.onnx`
- `vocab.txt`
- Configuration files

You can download pre-converted ONNX models or convert from NeMo format.

## Usage

### Quick Start (Auto-Download TDT Model)

The easiest way to get started is to just run the app without any arguments:

```bash
./parakeet
```

If no model is found, the app will offer to automatically download the TDT multilingual model (~3GB) from Hugging Face to `models/parakeet-tdt/`.

### Basic Usage (CTC Model)

```bash
# Use model from current directory
./parakeet --model . --no-tdt

# Or specify a model file
./parakeet --model /path/to/model.onnx --no-tdt
```

### TDT Model (Multilingual)

```bash
# Use TDT model (default)
./parakeet --model /path/to/tdt_model_dir --tdt

# Or just use default location (will download if missing)
./parakeet
```

### Command Line Options

```
Options:
  -m, --model <MODEL>              Path to ONNX model directory or file
                                   [default: models/parakeet-tdt, will auto-download if missing]
  -t, --tdt                        Use TDT model (multilingual) [default: true]
  -o, --output-dir <DIR>           Output directory for transcripts [default: transcripts]
  -k, --keep-audio                 Keep audio recordings (don't delete after transcription)
  -s, --sample-rate <SAMPLE_RATE>  Audio sample rate in Hz [default: 16000]
  -c, --config-file <FILE>         Path to config file [default: config.toml]
  -h, --help                       Print help
  -V, --version                    Print version
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

### English Transcription (CTC)
```bash
./parakeet --model ./parakeet_ctc
```

### Multilingual Transcription (TDT)
```bash
./parakeet --model ./parakeet_tdt --tdt
```

### Keep Audio Files
```bash
./parakeet --model ./model --keep-audio
```

### Custom Output Directory
```bash
./parakeet --model ./model --output-dir ~/my_transcripts
```

### Custom Sample Rate
```bash
# Use 8kHz sample rate (e.g., for phone call audio)
./parakeet --model ./model --sample-rate 8000

# Default is 16kHz (recommended for Parakeet models)
./parakeet --model ./model --sample-rate 16000
```

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
