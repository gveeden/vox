```markdown
# Parakeet Voice Transcription

A voice-to-text transcription tool using NVIDIA's Parakeet ASR model with local hotkey support for macOS.

## What is parakeet.py?

`parakeet.py` is a very basic (WIP) real-time voice transcription application that allows you to record audio using a keyboard shortcut and automatically transcribe it to text. The transcribed text is automatically pasted at your cursor location.

Currently just for my personal use.

### Features

- **Local ASR Model**: Uses NVIDIA's Parakeet CTC 0.6B model for high-quality speech recognition
- **Punctuation & Capitalization**: Automatically adds proper punctuation and capitalization using BERT
- **Hotkey Control**: Hold `Command+Option+Space` to record, release `Space` to transcribe
- **Visual Feedback**: Shows a recording overlay while capturing audio
- **Auto-Paste**: Transcribed text is automatically copied to clipboard and pasted at cursor
- **Transcript Storage**: Saves all transcripts to the `transcripts/` folder with timestamps

## Setup

### Prerequisites

- Python 3.8+
- macOS

### Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv speech_to_text_env
source speech_to_text_env/bin/activate
```

2. Install dependencies:
```bash
pip install sounddevice scipy pynput numpy pyperclip pyautogui nemo_toolkit[asr] PyObjC
```

## Usage

1. Run the script:
```bash
python parakeet.py
```

2. Wait for models to load (first run may take longer)

3. Hold `Command+Option+Space` to start recording
   - A recording indicator overlay will appear

4. Speak your text

5. Release `Space` to stop recording and transcribe
   - The transcribed text will be automatically pasted at your cursor location
   - Transcript is saved to `transcripts/transcript_YYYY-MM-DD_HH-MM-SS.txt`

## How It Works

1. **Audio Recording**: Captures audio at 16kHz using sounddevice
2. **ASR Processing**: Transcribes audio using NVIDIA Parakeet CTC 0.6B model
3. **Post-Processing**: Adds punctuation and capitalization using BERT-based model
4. **Output**: Copies to clipboard and auto-pastes to cursor location

## Files

- `parakeet.py` - Main transcription application
- `overlay.py` - Visual recording indicator using PyObjC
- `transcripts/` - Directory containing all saved transcripts (auto-created)

## Notes

- Audio recordings are temporarily saved as `.wav` files and deleted after transcription
- Requires microphone permissions on macOS
- The application runs as a background listener until terminated (Ctrl+C)
