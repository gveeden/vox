# Parakeet CLI - Voice Transcription Tool

A Rust implementation of voice transcription using NVIDIA Parakeet ASR models via ONNX Runtime. Record audio with a keyboard shortcut and automatically transcribe to text.

## Features

- 🎙️ Voice recording with keyboard hotkey (Cmd+Option+Space)
- 🤖 Support for any ONNX-format Parakeet models
- 🌍 CTC (English-only) and TDT (multilingual, 25+ languages) models
- 📋 Automatic clipboard copy and paste
- 💾 Save transcripts to files
- 🔔 Native notifications
- ⚡ Fast inference using ONNX Runtime
- 🖥️ Cross-platform (macOS, Linux, Windows)

## Requirements

- Rust 1.70 or later
- ONNX model files (see Model Setup below)
- Microphone access

## Installation

```bash
cargo build --release
```

The binary will be available at `target/release/parakeet`.

### Running as a Background Service (macOS)

To run CleverNote automatically on login:

```bash
./install_launchagent.sh
```

This will:
- Build the release binary
- Install the LaunchAgent plist
- Start the service automatically

**Important:** You must grant Accessibility permissions to the binary for hotkeys and pasting to work. See [LAUNCHAGENT_SETUP.md](LAUNCHAGENT_SETUP.md) for detailed instructions.

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

Create a `config.toml` file to customize hotkeys:

```toml
# Modifier key: Alt, Ctrl, Cmd, or Shift
modifier_key = "Alt"

# Trigger key: Space, Return, Tab, Escape
trigger_key = "Space"
```

The application will create a default config.toml on first run if one doesn't exist.

## How to Use

1. Start the application:
   ```bash
   ./parakeet
   ```
   
   If no model is found, you'll be prompted to download one automatically.

2. Wait for the "Ready!" message

3. Press **Alt+Space** (default) to start recording
   - Microphone activates only while recording (not running in background)
   - **Hotkey is captured and won't reach other apps** (no stray spaces!)

4. Speak your message

5. Press **Alt+Space** again to stop recording and transcribe

6. The transcribed text will be:
   - Displayed in the terminal
   - Saved to `transcripts/transcript_TIMESTAMP.txt`
   - Copied to clipboard
   - Automatically pasted at cursor position

7. Press **Alt+Space** again for another recording - works immediately!

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
