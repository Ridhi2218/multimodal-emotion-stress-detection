# ============================================================
#  VOICE MODULE — Records voice and extracts MFCC features
#  This is the "Speech Emotion Analysis Module" from your paper
#  Uses LSTM-ready feature extraction (MFCC)
# ============================================================

import numpy as np
import librosa
import sounddevice as sd
import os


# ── Emotion labels matching RAVDESS dataset ──────────────────
EMOTIONS = ['neutral', 'calm', 'happy', 'sad',
            'angry', 'fearful', 'disgust', 'surprised']


def record_audio(duration=3, sample_rate=22050):
    """
    Records audio from microphone for given duration.
    
    duration    : seconds to record (default 3)
    sample_rate : audio quality — 22050 is standard
    
    Returns: numpy array of audio samples
    """
    print(f"🎤 Recording for {duration} seconds... Speak now!")
    
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # Wait until recording is finished
    print("✅ Recording complete.")
    
    return recording.flatten(), sample_rate


def extract_mfcc_features(audio, sample_rate=22050, n_mfcc=40):
    """
    Extracts MFCC (Mel-Frequency Cepstral Coefficients) from audio.
    
    This is EXACTLY the feature extraction technique described in
    your research paper under "Speech Emotion Recognition".
    
    MFCC converts raw audio into 40 meaningful numbers that
    capture the tone, pitch, and rhythm of speech.
    
    Returns: numpy array of shape (40,)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    
    # Take mean across time — gives us one value per MFCC coefficient
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    return mfcc_mean


def extract_full_features(audio, sample_rate=22050):
    """
    Extracts a richer set of audio features:
    - MFCC (tone/pitch)
    - Chroma (harmonic content)
    - Spectral contrast (loudness variation)
    
    Combined into one feature vector for better accuracy.
    Returns: numpy array of shape (193,)
    """
    # MFCC — 40 features
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_features = np.mean(mfcc.T, axis=0)

    # Chroma — 12 features (musical pitch classes)
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma_features = np.mean(chroma.T, axis=0)

    # Mel spectrogram — 128 features
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
    mel_features = np.mean(mel.T, axis=0)

    # Spectral contrast — 7 features
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    contrast_features = np.mean(contrast.T, axis=0)

    # Tonnetz — 6 features
    harmonic = librosa.effects.harmonic(audio)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
    tonnetz_features = np.mean(tonnetz.T, axis=0)

    # Concatenate all features
    all_features = np.concatenate([
        mfcc_features,
        chroma_features,
        mel_features,
        contrast_features,
        tonnetz_features
    ])

    return all_features


def analyze_voice_energy(audio):
    """
    Simple rule-based voice stress indicator based on energy.
    High energy + high pitch variation → more likely stressed.
    
    Returns: stress score between 0 and 1
    """
    # RMS energy (loudness)
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Zero crossing rate (how often signal changes sign — relates to pitch)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

    # Normalize to 0-1 range (approximate)
    energy_score = min(rms * 10, 1.0)
    zcr_score = min(zcr * 5, 1.0)

    stress_indicator = (energy_score * 0.6) + (zcr_score * 0.4)
    return round(stress_indicator, 3)


def load_audio_file(filepath):
    """
    Load audio from a file instead of recording.
    Useful for testing without a microphone.
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None, None
    
    audio, sr = librosa.load(filepath, duration=3)
    return audio, sr


# Run directly to test this module alone
if __name__ == "__main__":
    print("Testing Voice Module...")
    audio, sr = record_audio(duration=3)
    
    features = extract_mfcc_features(audio, sr)
    print(f"✅ MFCC extracted: {len(features)} features")
    print(f"   First 5 values: {features[:5].round(2)}")
    
    stress = analyze_voice_energy(audio)
    print(f"   Voice stress indicator: {stress:.2%}")
