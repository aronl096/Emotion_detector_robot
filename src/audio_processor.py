import sounddevice as sd
import numpy as np

def record_audio(duration, sample_rate, device=None):
    print("Recording... Please speak into the microphone.")
    recording = sd.rec(
        int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype="float32", device=device
    )
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return recording.flatten()
