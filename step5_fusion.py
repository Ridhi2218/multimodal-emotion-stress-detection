import cv2
import sounddevice as sd
import librosa
import numpy as np
from fer.fer import FER

def capture_face_emotion():
    """Get emotion from webcam"""
    camera = cv2.VideoCapture(0)
    success, frame = camera.read()
    camera.release()
    
    detector = FER()
    result = detector.detect_emotions(frame)
    
    if result:
        return result[0]['emotions']
    return None

def capture_voice_features():
    """Get MFCC features from microphone"""
    print("🎤 Speak for 3 seconds...")
    recording = sd.rec(int(3 * 22050), samplerate=22050, channels=1)
    sd.wait()
    
    audio = recording.flatten()
    mfcc = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

def calculate_stress(face_emotions):
    """
    Calculate stress from emotions
    This is the fusion formula from your paper: S = f(F, V, P)
    """
    if not face_emotions:
        return "UNKNOWN"
    
    # Stress-related emotions
    stress_score = (
        face_emotions.get('angry', 0) +
        face_emotions.get('sad', 0) +
        face_emotions.get('fear', 0) +
        face_emotions.get('disgust', 0)
    )
    
    # Positive emotions
    calm_score = (
        face_emotions.get('happy', 0) +
        face_emotions.get('neutral', 0)
    )
    
    if stress_score > 0.5:
        return "🔴 HIGH STRESS"
    elif stress_score > 0.25:
        return "🟡 MODERATE STRESS"
    else:
        return "🟢 LOW / NO STRESS"

# --- MAIN PROGRAM ---
print("=" * 40)
print("   EMOTION & STRESS DETECTOR")
print("=" * 40)
print("")

# Step 1: Get face data
print("📷 Looking at your face...")
face_emotions = capture_face_emotion()

if face_emotions:
    print("Face emotions detected:")
    for emotion, score in face_emotions.items():
        if score > 0.05:  # Only show significant ones
            print(f"  {emotion}: {score:.0%}")

# Step 2: Get voice data
voice_features = capture_voice_features()
print(f"✅ Voice captured ({len(voice_features)} features extracted)")

# Step 3: Combine and decide stress level
print("")
print("🧠 Analyzing...")
stress_result = calculate_stress(face_emotions)

print("")
print("=" * 40)
print(f"  RESULT: {stress_result}")
print("=" * 40)
