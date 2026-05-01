import sounddevice as sd
import librosa
import numpy as np

print("Recording your voice for 3 seconds...")
print("Say something now!")

# Record 3 seconds of audio
# 22050 = audio quality (samples per second)
duration = 3
sample_rate = 22050
recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)

# Wait for recording to finish
sd.wait()
print("Recording done!")

# Convert recording to 1D array
audio = recording.flatten()

# Extract MFCC features
# MFCC = 40 numbers that describe your voice
# Your paper mentions this technique
mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfcc_features = np.mean(mfcc.T, axis=0)

print(f"Voice converted to {len(mfcc_features)} numbers (features)")
print("First 5 features:", mfcc_features[:5].round(2))
print("✅ Voice processing works!")
