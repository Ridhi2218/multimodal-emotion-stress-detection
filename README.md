# Real-Time Multimodal Emotion and Stress Detection System

## 📋 Project Overview

This is a comprehensive **AI/ML project** that detects human emotions and stress levels in real-time by combining:
- **Facial Recognition (CNN)** - Analyzes facial expressions
- **Speech Analysis (LSTM)** - Analyzes vocal patterns
- **Multimodal Fusion** - Combines both modalities for accurate stress prediction

**Paper Reference**: Real-Time Multimodal Emotion and Stress Detection System

---

## 🎯 Key Features

✅ **Real-Time Analysis** - Live webcam and microphone input
✅ **7-Class Emotion Detection** - Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral
✅ **Stress Classification** - LOW / MODERATE / HIGH stress levels
✅ **Interactive Dashboard** - Streamlit web interface
✅ **Pre-Trained Models** - CNN via FER library, LSTM trainable
✅ **Multimodal Fusion** - Weighted combination of face + voice signals
✅ **Personalized Recommendations** - Stress management advice

---

## 🏗️ Project Architecture

```
emotion_project/
├── app.py                    # Main Streamlit dashboard
├── train_lstm.py             # LSTM training script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── src/                      # Source code modules
│   ├── config.py            # Configuration & hyperparameters
│   ├── face_module.py       # CNN facial emotion detection
│   ├── voice_module.py      # LSTM speech emotion analysis
│   ├── fusion.py            # Multimodal fusion logic
│   └── utils.py             # Utility functions
│
├── data/                     # Datasets
│   ├── fer2013/             # Facial images
│   │   ├── train/           # Training images (35,000+)
│   │   └── validation/      # Validation images
│   └── ravdess/             # Speech audio files
│       ├── train/           # Training audio (1,500+)
│       └── test/            # Test audio
│
├── models/                   # Trained models
│   ├── lstm_voice_model.h5  # Trained LSTM model
│   ├── scaler.pkl           # MFCC scaler
│   └── cnn_face_model.h5    # Face model (pre-loaded)
│
└── outputs/                  # Analysis outputs
    └── sessions/            # Saved analysis results
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download project
cd emotion_project

# Install dependencies (first time only)
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start Streamlit dashboard
streamlit run app.py

# Browser will open at http://localhost:8501
```

### 3. Use the Application

1. Click "🔍 Analyze Emotions"
2. Position your face in front of the webcam
3. Speak for 3 seconds when prompted
4. View your emotion analysis and stress level

---

## 📊 How It Works

### Step 1: Facial Recognition (CNN)

```
📷 Webcam Input
      ↓
[Pre-trained FER Model]
(Facial Expression Recognition)
      ↓
7 Emotion Scores: {happy: 0.85, sad: 0.05, ...}
```

- Uses **Convolutional Neural Network (CNN)**
- Pre-trained on **FER-2013 dataset** (35,000+ labeled images)
- Detects facial micro-expressions
- Outputs probability for each emotion

### Step 2: Speech Analysis (LSTM)

```
🎤 Microphone Input
      ↓
[MFCC Feature Extraction]
(40 Mel Frequency Cepstral Coefficients)
      ↓
[LSTM Model]
(Learns temporal patterns in speech)
      ↓
Stress Score: 0.0-1.0
```

- Extracts **40 MFCC features** from audio
- Captures: pitch, tone, rhythm, energy
- **LSTM** learns temporal dependencies
- Indicates stress through vocal patterns

**Key Voice Indicators:**
- High energy + rapid pitch changes = **HIGH STRESS**
- Low energy + stable pitch = **CALM**

### Step 3: Multimodal Fusion

**Formula from Paper**: `S = f(F, V, P)`

```
Final Stress Score = (0.60 × Face Score) + (0.40 × Voice Score)

Where:
F = Facial Emotions
V = Voice Features  
P = Physiological Signals (future work)
```

**Why these weights?**
- Face is more reliable (harder to fake expressions)
- Voice is supplementary but contains valuable info

### Step 4: Stress Classification

```
Stress Emotions Score:
  = angry + disgust + fear + sad

If score > 0.60  → 🔴 HIGH STRESS
If score > 0.30  → 🟡 MODERATE STRESS
If score ≤ 0.30  → 🟢 LOW STRESS
```

---

## 🎓 Understanding the ML Concepts

### CNN (Convolutional Neural Network) for Faces

**What it does:**
1. Input: 48×48 grayscale face image
2. **Convolution**: Detects edges, textures
3. **Pooling**: Reduces dimensionality
4. **Flattening**: Converts to vector
5. **Dense**: Classifies to 7 emotions

**Why CNN?**
- Excellent at spatial pattern recognition
- Pre-trained models available
- Real-time inference (fast)

### LSTM (Long Short-Term Memory) for Speech

**What it does:**
1. Input: 40 MFCC features per frame
2. **LSTM Cell**: Remembers previous context
3. **Gates**: Learn what to remember/forget
4. **Output**: Emotion classification

**Why LSTM?**
- Speech is temporal (sequence matters)
- Captures long-term dependencies
- Better than simple RNN for speech

### MFCC (Mel Frequency Cepstral Coefficients)

**What it is:**
- 40 numbers representing acoustic characteristics
- Based on human hearing perception
- Similar to what our ears perceive

**Calculation:**
```
Raw Audio → FFT → Mel Filterbank → Log → DCT → 40 MFCC
```

### Fusion Strategy

**Simple but Powerful:**
```python
fused = 0.6 * face_emotion + 0.4 * voice_stress
```

**Advantages:**
- Reduces error from single modality
- Face + Voice gives complete picture
- Easily adjustable weights (in app sidebar)

---

## 📈 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                   │
│                        app.py                            │
└──────────────┬──────────────────────────────────────────┘
               │
      ┌────────┴────────┐
      │                 │
      ↓                 ↓
┌──────────────┐  ┌──────────────┐
│ face_module  │  │ voice_module │
│    (CNN)     │  │   (MFCC +    │
│              │  │    LSTM)     │
└──────┬───────┘  └────────┬─────┘
       │                   │
       └───────────┬───────┘
                   ↓
            ┌─────────────┐
            │  fusion.py  │
            │  (Weighted  │
            │  Combine)   │
            └──────┬──────┘
                   ↓
        ┌──────────────────────┐
        │  Stress Classification│
        │  + Recommendations   │
        └──────────────────────┘
```

---

## 🔧 Configuration

All settings are in `src/config.py`:

```python
# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Audio settings
AUDIO_SAMPLE_RATE = 22050 Hz
AUDIO_DURATION = 3 seconds
N_MFCC = 40 features

# Fusion weights
FUSION_WEIGHTS = {'face': 0.60, 'voice': 0.40}

# Stress thresholds
STRESS_THRESHOLDS = {'high': 0.60, 'moderate': 0.30}
```

---

## 📥 Datasets Used

| Component | Dataset | Size | Source |
|-----------|---------|------|--------|
| **Face** | FER-2013 | 35,887 images | [Kaggle](https://www.kaggle.com/msambare/fer2013) |
| **Voice** | RAVDESS | 1,500 audio files | [Zenodo](https://zenodo.org/record/1188976) |
| **Physiological** | WESAD | 600 sessions | [Dataset Link](http://ubicomp.eti.uni-siegen.de/home/datasets/wesad/) |

### Download Instructions

**FER-2013 (Facial Images):**
1. Go to Kaggle: https://www.kaggle.com/msambare/fer2013
2. Download CSV file
3. Extract to `data/fer2013/`

**RAVDESS (Speech Audio):**
1. Go to Zenodo: https://zenodo.org/record/1188976
2. Download all actor folders
3. Extract to `data/ravdess/train/`

---

## 🤖 Model Details

### Face Detection Model (FER)

- **Architecture**: Pre-trained CNN
- **Training Data**: FER-2013 (35,887 images)
- **Classes**: 7 emotions
- **Accuracy**: ~65-75% (public benchmark)
- **Library**: python-fer (uses MTCNN for face detection)

### Voice Detection Model (LSTM)

```
Input Layer (120 features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Output Layer (7 emotions, Softmax)
```

**Training:**
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical Crossentropy
- Epochs: 50
- Batch Size: 32
- Early Stopping: Yes

---

## 🎯 Performance Metrics

The project aims for:
- **Face Accuracy**: 70%+
- **Voice Accuracy**: 65%+
- **Fused Accuracy**: 75%+ (multimodal advantage)
- **Inference Time**: < 2 seconds per analysis

---

## 💻 System Requirements

**Minimum:**
- Windows 10 / Mac / Linux
- Python 3.8+
- 4GB RAM
- Webcam + Microphone

**Recommended:**
- Python 3.10+
- 8GB RAM
- GPU (NVIDIA with CUDA)
- USB microphone for better audio

---

## 🚨 Troubleshooting

### "Camera not found"
- Check webcam is connected
- Try closing other apps using webcam
- Restart terminal and try again

### "No face detected"
- Ensure good lighting
- Position face directly toward camera
- Remove glasses if possible
- Check camera isn't blocked

### "Failed to record audio"
- Check microphone is connected
- Run: `pip install pyaudio`
- Restart Streamlit app

### "Module not found" error
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python path: `which python`
- Create virtual environment (recommended)

### Slow performance
- Close other applications
- Use GPU (if available): Check TensorFlow setup
- Reduce analysis frequency
- Ensure good lighting (reduces processing)

---

## 🔬 Experiments You Can Run

### 1. Test Different Emotions
```python
# Try each emotion and see accuracy
emotions_to_test = ['happy', 'angry', 'sad', 'neutral', ...]
```

### 2. Adjust Fusion Weights
Use the sidebar slider in Streamlit to adjust face/voice weights and see impact on stress classification.

### 3. Train LSTM on Custom Data
```bash
python train_lstm.py
```

### 4. Compare Single vs Multimodal
- Analyze with face only
- Analyze with voice only
- Compare with fusion result

---

## 📚 References & Papers

1. **Facial Expression Recognition**
   - Goodfellow et al., "Challenges in Representation Learning: A report on three machine learning contests"
   - FER-2013 Dataset Paper

2. **Speech Emotion Recognition**
   - Livingstone et al., "The RAVDESS Emotional Speech and Song Database"
   - MFCC Feature Extraction: Davis & Mermelstein, "Comparison of Parametric Representations for Monosyllabic Word Recognition"

3. **Multimodal Emotion Recognition**
   - Poria et al., "Multimodal Emotion Recognition on IEMOCAP Dataset using Deep Learning"
   - Baltrušaitis et al., "Multimodal Machine Learning: A Survey and Taxonomy"

4. **LSTM for Sequence Modeling**
   - Hochreiter & Schmidhuber, "Long Short-Term Memory"
   - Graves, "Supervised Sequence Labelling with Recurrent Neural Networks"

---

## 📝 Project Development Steps

### Week 1: Setup & Face Detection ✓
- [x] Install libraries
- [x] Create face_module.py
- [x] Test webcam capture
- [x] Integrate FER model

### Week 2: Speech Analysis ✓
- [x] Create voice_module.py
- [x] MFCC feature extraction
- [x] Voice stress analysis
- [x] Test microphone

### Week 3: Fusion Module ✓
- [x] Create fusion.py
- [x] Implement weighted fusion
- [x] Stress classification logic
- [x] Recommendation system

### Week 4: Dashboard ✓
- [x] Create Streamlit app
- [x] Add real-time analysis
- [x] Visualization
- [x] Configuration controls

### Week 5: Training ✓
- [x] Create train_lstm.py
- [x] LSTM model architecture
- [x] Training pipeline
- [x] Model evaluation

### Week 6: Production Ready ✓
- [x] Error handling
- [x] Documentation
- [x] Configuration management
- [x] Performance optimization

---

## 🎁 Future Enhancements

- [ ] Physiological signals (heart rate, skin conductance)
- [ ] Real-time video recording and playback
- [ ] Historical analysis trends
- [ ] Multi-person detection
- [ ] Emotion intensity tracking
- [ ] Custom model training UI
- [ ] Mobile app deployment
- [ ] Cloud integration (AWS/Azure)
- [ ] Advanced visualization (3D emotion space)
- [ ] Export reports and statistics

---

## 📄 License

This project is for educational purposes. Feel free to modify and distribute.

---

## 👨‍💻 Author & Contact

Created for: **Multimodal Emotion and Stress Detection AI/ML Project**

For questions or improvements, refer to the documentation and code comments.

---

## 🙏 Acknowledgments

- **FER Library**: Contributors to the python-fer package
- **Librosa**: Audio analysis library
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Dashboard framework
- **Research Community**: RAVDESS, FER-2013, WESAD dataset creators

---

**Last Updated**: April 25, 2026

**Status**: ✅ Production Ready
