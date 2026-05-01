# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- Webcam
- Microphone
- ~2GB free disk space

### Step 1: Initial Setup (Do this once)

```bash
# Open terminal in project folder
cd emotion_project

# Run setup script
python setup.py

# This will:
# ✓ Create folder structure
# ✓ Install all dependencies
# ✓ Verify installations
```

### Step 2: Download Datasets (Optional)

For training your own models:

**Option A: Download FER-2013 (Faces)**
1. Visit: https://www.kaggle.com/msambare/fer2013
2. Download `fer2013.csv`
3. Extract images to `data/fer2013/train` and `data/fer2013/validation`

**Option B: Download RAVDESS (Voice)**
1. Visit: https://zenodo.org/record/1188976
2. Download actor folders
3. Extract to `data/ravdess/train/[emotion]/`

### Step 3: Start Using the App

```bash
# Start the Streamlit dashboard
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

### Step 4: Analyze Your Emotions

1. **Position yourself**: Sit in front of your webcam
2. **Ensure lighting**: Make sure your face is well-lit
3. **Click "Analyze Emotions"**: Button in the app
4. **Speak for 3 seconds**: When the app asks for voice input
5. **View results**: See your emotion analysis and stress level

---

## 🎯 What You'll See

### Output Display

```
RESULT PAGE
├── 🎯 Stress Level: [LOW/MODERATE/HIGH STRESS]
├── ✨ Confidence: 92%
├── 🧠 Dominant Emotion: Happy (92%)
│
├── 📊 Fused Emotions
│   └── Bar chart of all 7 emotions
│
├── 📈 Emotion Breakdown
│   ├── Stress Emotions: Angry, Fear, Sad, Disgust
│   └── Calm Emotions: Happy, Neutral
│
└── 💡 Recommendations
    └── Personalized advice based on stress level
```

---

## 📋 Module-by-Module Usage

### Use Case 1: Just Test Face Recognition

```python
from src.face_module import create_face_detector

detector = create_face_detector()
frame = detector.capture_photo()
emotions = detector.detect_emotions(frame)
print(emotions)
```

### Use Case 2: Just Test Voice Analysis

```python
from src.voice_module import create_speech_analyzer

analyzer = create_speech_analyzer()
audio, sr = analyzer.record_audio(duration=3)
mfcc = analyzer.extract_mfcc_features(audio, sr)
stress = analyzer.extract_voice_stress_indicators(audio, sr)
print(f"Stress Score: {stress['stress_score']}")
```

### Use Case 3: Just Test Fusion

```python
from src.fusion import create_fusion_module

fusion = create_fusion_module(face_weight=0.6, voice_weight=0.4)

# Sample data
face_emotions = {'happy': 0.8, 'sad': 0.1, ...}
voice_stress = 0.3

result = fusion.full_analysis(face_emotions, voice_stress)
print(result['stress_level'])  # Output: 'LOW STRESS'
```

---

## 🔧 Troubleshooting

### Problem: "Camera not found"
```bash
# Solution 1: Check device manager
# Ensure webcam is connected and not in use

# Solution 2: Restart Streamlit
streamlit run app.py

# Solution 3: Try another app (like Photo app) to verify camera works
```

### Problem: "No face detected"
- ✓ Ensure good lighting
- ✓ Remove sunglasses
- ✓ Face camera directly
- ✓ Position face in center of frame

### Problem: "Failed to record audio"
```bash
# Solution 1: Install audio dependencies
pip install pyaudio

# Solution 2: Check microphone in system settings
# Windows: Settings → Sound → Input devices
# Mac: System Preferences → Sound → Input
# Linux: pavucontrol (install if needed)
```

### Problem: "Module not found" error
```bash
# Solution: Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or create fresh virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 💡 Tips for Best Results

### For Facial Recognition:
- ✅ Face camera directly (not at angle)
- ✅ Good lighting from front
- ✅ Full face visible (not partially cut off)
- ✅ Natural facial expression (5-10 seconds)

### For Voice Analysis:
- ✅ Quiet room (no background noise)
- ✅ Speak clearly and naturally
- ✅ Vary your tone/emotion
- ✅ 3-5 seconds of continuous speech

### For Better Accuracy:
- ✅ Use external USB microphone
- ✅ Analyze multiple times for average
- ✅ Try different lighting conditions
- ✅ Adjust fusion weights based on your situation

---

## 🎓 Learning Resources

### Understanding the Project:
1. Read `README.md` - Complete project documentation
2. Read inline code comments - Each module is well documented
3. Check `src/config.py` - All settings and parameters

### Understanding the ML:
- **CNN for Faces**: See `src/face_module.py` comments
- **LSTM for Voice**: See `src/voice_module.py` comments
- **Fusion Logic**: See `src/fusion.py` comments

### Training Your Own Model:
```bash
# Download RAVDESS dataset first
# Then run:
python train_lstm.py

# This trains an LSTM model on speech emotions
# Saves to: models/lstm_voice_model.h5
```

---

## 🚀 Advanced Usage

### Change Fusion Weights

In Streamlit app:
1. Use sidebar slider
2. Adjust "Face Importance"
3. See how predictions change

**Example:**
- Face 80%, Voice 20% → Relies more on expressions
- Face 40%, Voice 60% → Relies more on voice tone

### Export Results

```python
import json
from src.utils import FileManager

# Save results
result = {...}  # From analysis
FileManager.save_json(result, "my_analysis.json")

# Load results
loaded = FileManager.load_json("my_analysis.json")
```

### Batch Analysis

```python
from src.face_module import create_face_detector
from src.voice_module import create_speech_analyzer
from src.fusion import create_fusion_module

detector = create_face_detector()
analyzer = create_speech_analyzer()
fusion = create_fusion_module()

# Analyze multiple times and average results
results = []
for i in range(3):
    face_emo = detector.detect_emotions(...)
    voice_stress = analyzer.extract_voice_stress_indicators(...)
    result = fusion.full_analysis(face_emo, voice_stress)
    results.append(result)

# Average the results
avg_stress = sum(r['stress_score'] for r in results) / len(results)
```

---

## 📞 Support

If you encounter issues:

1. Check README.md for full documentation
2. Review code comments in relevant module
3. Check that all dependencies are installed
4. Ensure webcam/microphone work in other apps
5. Try in a different location with better lighting

---

## ✨ Next Steps After First Use

1. **Download datasets** - Get FER-2013 and RAVDESS
2. **Train LSTM model** - `python train_lstm.py`
3. **Experiment with weights** - Try different fusion weights
4. **Try multiple analyses** - See consistency
5. **Extend the project** - Add more features!

---

**Ready? Start with:**
```bash
streamlit run app.py
```

Happy analyzing! 🧠✨
