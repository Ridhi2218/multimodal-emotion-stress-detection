# ============================================================
#  VOICE MODULE — Records audio and extracts MFCC features
#  This is the "Speech Emotion Recognition Module" from your paper
# ============================================================

import sounddevice as sd
import librosa
import numpy as np


def record_audio(duration=3, sr=22050):
    """
    Records audio from the microphone.
    Returns: (audio_array, sample_rate)
    """
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = recording.flatten()
    return audio, sr


def extract_mfcc_features(audio, sr=22050, n_mfcc=40):
    """
    Extracts MFCC features from audio.
    Returns: numpy array of 40 features
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)


def analyze_voice_energy(audio):
    """
    Calculates a normalised voice energy/stress score (0.0 – 1.0).
    Higher energy → higher stress score.
    """
    if audio is None or len(audio) == 0:
        return 0.0
    rms = np.sqrt(np.mean(audio ** 2))
    # Normalise to 0-1 range (typical RMS for speech ~0.01–0.3)
    score = float(np.clip(rms * 5, 0.0, 1.0))
    return score


if __name__ == "__main__":
    print("Recording 3 seconds of audio...")
    audio, sr = record_audio(duration=3)
    features = extract_mfcc_features(audio, sr)
    energy = analyze_voice_energy(audio)
    print(f"MFCC features extracted: {len(features)}")
    print(f"First 5 features: {features[:5].round(2)}")
    print(f"Voice energy score: {energy:.2%}")
    print("✅ Voice module works!")
