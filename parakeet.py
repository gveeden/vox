import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import threading
import pynput
import numpy as np
import pyperclip
import pyautogui
import nemo.collections.asr as nemo_asr

from nemo.collections.nlp.models import PunctuationCapitalizationModel
import os
import subprocess
import sys

# ✅ Load your Parakeet ASR model (can swap with "stt_en_conformer_transducer_large" if you want)
print("🔊 Loading ASR model (this may take a bit)...")
MODEL_NAME = "nvidia/parakeet-ctc-0.6b"
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
print("✅ ASR Model loaded!")

# Load punctuation and capitalization model
print("🔤 Loading punctuation model...")
punct_model = PunctuationCapitalizationModel.from_pretrained('punctuation_en_bert')
print("✅ Punctuation model loaded!")

recording = []
is_recording = False
record_thread = None
fs = 16000  # Sample rate

def record_audio():
    """Continuously record audio while is_recording is True"""
    global recording
    print("🎙️ Recording... Speak now.")
    rec_data = []

    with sd.InputStream(samplerate=fs, channels=1, dtype='float32') as stream:
        while is_recording:
            data, _ = stream.read(1024)
            rec_data.append(data)
    recording = np.concatenate(rec_data, axis=0)
    print("🛑 Recording stopped. Transcribing...")

    # Save the audio
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    audio_path = f"recording_{timestamp}.wav"
    write(audio_path, fs, recording)

    # Transcribe
    try:
        text_output = asr_model.transcribe([audio_path])
        # Extract text from Hypothesis object
        if isinstance(text_output, list) and len(text_output) > 0:
            text = text_output[0].text if hasattr(text_output[0], 'text') else str(text_output[0])
        else:
            text = text_output.text if hasattr(text_output, 'text') else str(text_output)
        
        # Add punctuation and capitalization
        text = punct_model.add_punctuation_capitalization([text])[0]
        
        # Create transcripts folder if it doesn't exist
        os.makedirs("transcripts", exist_ok=True)
        
        transcript_path = f"transcripts/transcript_{timestamp}.txt"
        with open(transcript_path, "w") as f:
            f.write(text)
        
        # Remove the audio recording
        os.remove(audio_path)

        # Copy and paste to cursor location
        pyperclip.copy(text)
        import time
        time.sleep(0.5)  # Give user time to position cursor
        pyautogui.hotkey("command", "v")

        print(f"💬 Transcript: {text[:100]}...")
        print(f"💾 Saved to {transcript_path}")
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        show_notification(f"❌ Transcription failed: {e}")

cmd_pressed = False
opt_pressed = False
overlay_process = None

def show_overlay():
    """Show the recording overlay"""
    global overlay_process
    if overlay_process is None:
        overlay_process = subprocess.Popen(
            [sys.executable, 'overlay.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

def hide_overlay():
    """Hide the recording overlay"""
    global overlay_process
    if overlay_process:
        try:
            overlay_process.stdin.close()
            overlay_process.wait(timeout=1)
        except:
            overlay_process.terminate()
        overlay_process = None

def show_notification(message):
    """Show a macOS notification using osascript"""
    try:
        subprocess.run([
            'osascript', '-e',
            f'display notification "{message}" with title "Voice Transcription"'
        ], check=False)
    except Exception as e:
        print(f"Notification: {message}")

def on_press(key):
    global is_recording, record_thread, cmd_pressed, opt_pressed
    try:
        if key == pynput.keyboard.Key.cmd:
            cmd_pressed = True
        if key == pynput.keyboard.Key.alt:  # Option key is 'alt' in pynput
            opt_pressed = True
        if key == pynput.keyboard.Key.space and cmd_pressed and opt_pressed and not is_recording:
            is_recording = True
            show_overlay()
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()
    except Exception as e:
        print("Error on_press:", e)

def on_release(key):
    global is_recording, cmd_pressed, opt_pressed
    try:
        if key == pynput.keyboard.Key.cmd:
            cmd_pressed = False
        if key == pynput.keyboard.Key.alt:
            opt_pressed = False
        if key == pynput.keyboard.Key.space and is_recording:
            is_recording = False
            hide_overlay()
    except Exception as e:
        print("Error on_release:", e)

# Start hotkey listener
print("🚀 Ready! Hold Command+Option+Space to record. Release Space to transcribe.")
with pynput.keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
