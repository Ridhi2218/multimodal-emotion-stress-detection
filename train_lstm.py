"""
LSTM Model Training Script for Speech Emotion Recognition
Trains on RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song)
Extracts MFCC features and trains LSTM for 7-class emotion classification
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List

import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    N_MFCC, AUDIO_SAMPLE_RATE, N_FFT, HOP_LENGTH,
    LSTM_UNITS_LAYER1, LSTM_UNITS_LAYER2, LSTM_DROPOUT,
    LSTM_DENSE_UNITS, LSTM_BATCH_SIZE, LSTM_EPOCHS,
    EMOTION_LIST
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    Trains LSTM model for speech emotion recognition
    
    Architecture:
    - Input: 40 MFCC features
    - LSTM Layer 1: 128 units
    - Dropout: 0.3
    - LSTM Layer 2: 64 units
    - Dense Layer: 32 units
    - Output: 7 emotions (softmax)
    """
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
        logger.info("LSTM Trainer initialized")
    
    def extract_mfcc_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MFCC features array of shape (N_MFCC,)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE)
            
            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=N_MFCC,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH
            )
            
            # Mean across time
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            # Add delta features
            mfcc_delta = np.mean(librosa.feature.delta(mfcc).T, axis=0)
            mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
            
            # Combine
            features = np.concatenate([mfcc_mean, mfcc_delta, mfcc_delta2])
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract MFCC from {audio_path}: {e}")
            return None
    
    def prepare_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset from directory structure
        Expected structure:
        data_dir/
            emotion1/audio1.wav
            emotion1/audio2.wav
            emotion2/audio1.wav
            ...
        
        Args:
            data_dir: Root directory containing emotion subdirectories
            
        Returns:
            Tuple of (features_array, labels_array)
        """
        X = []
        y = []
        
        logger.info(f"Preparing dataset from {data_dir}")
        
        for emotion_idx, emotion in enumerate(EMOTION_LIST):
            emotion_dir = os.path.join(data_dir, emotion)
            
            if not os.path.isdir(emotion_dir):
                logger.warning(f"Emotion directory not found: {emotion_dir}")
                continue
            
            audio_files = [f for f in os.listdir(emotion_dir) 
                          if f.endswith('.wav')]
            
            logger.info(f"Processing {emotion}: {len(audio_files)} files")
            
            for audio_file in audio_files:
                audio_path = os.path.join(emotion_dir, audio_file)
                
                features = self.extract_mfcc_from_file(audio_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion_idx)
        
        if not X:
            logger.error("No audio files found in dataset")
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def build_model(self, input_shape: int = N_MFCC * 3) -> models.Model:
        """
        Build LSTM model for emotion classification
        
        Args:
            input_shape: Number of input features
            
        Returns:
            Compiled keras model
        """
        model = models.Sequential([
            # Reshape input for LSTM
            layers.Reshape((1, input_shape), input_shape=(input_shape,)),
            
            # LSTM Layer 1
            layers.LSTM(LSTM_UNITS_LAYER1, return_sequences=True),
            layers.Dropout(LSTM_DROPOUT),
            
            # LSTM Layer 2
            layers.LSTM(LSTM_UNITS_LAYER2, return_sequences=False),
            layers.Dropout(LSTM_DROPOUT),
            
            # Dense layers
            layers.Dense(LSTM_DENSE_UNITS, activation='relu'),
            layers.Dropout(LSTM_DROPOUT),
            
            # Output layer
            layers.Dense(len(EMOTION_LIST), activation='softmax')
        ])
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully")
        model.summary()
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             epochs: int = LSTM_EPOCHS) -> models.Model:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            
        Returns:
            Trained model
        """
        logger.info("Starting training...")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        else:
            X_val_scaled = None
        
        # Build model
        self.model = self.build_model(X_train_scaled.shape[1])
        
        # Prepare callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train
        validation_data = (X_val_scaled, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=LSTM_BATCH_SIZE,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
        
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with metrics
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        loss, accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        # Get predictions
        predictions = self.model.predict(X_test_scaled, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        report = classification_report(y_test, pred_classes, 
                                       target_names=EMOTION_LIST,
                                       output_dict=True)
        cm = confusion_matrix(y_test, pred_classes)
        
        metrics = {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def save_model(self, model_path: str = None, scaler_path: str = None):
        """
        Save model and scaler
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        if model_path is None:
            model_path = os.path.join(self.output_dir, 'lstm_voice_model.h5')
        if scaler_path is None:
            scaler_path = os.path.join(self.output_dir, 'scaler.pkl')
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str, scaler_path: str):
        """
        Load pretrained model
        
        Args:
            model_path: Path to model
            scaler_path: Path to scaler
        """
        self.model = keras.models.load_model(model_path)
        
        import pickle
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Model and scaler loaded successfully")


def train_on_ravdess(data_dir: str = "data/ravdess/train"):
    """
    Train LSTM on RAVDESS dataset
    
    Args:
        data_dir: Path to RAVDESS dataset
    """
    trainer = LSTMTrainer()
    
    # Prepare dataset
    X, y = trainer.prepare_dataset(data_dir)
    if X is None:
        logger.error("Failed to prepare dataset")
        return
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save
    trainer.save_model()
    
    logger.info("Training complete!")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    logger.info("LSTM Training Script for Speech Emotion Recognition")
    logger.info("Training on RAVDESS dataset...")
    
    data_path = "data/ravdess/train"
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset path not found: {data_path}")
        logger.info("Please download RAVDESS dataset first")
        logger.info("Dataset available at: https://zenodo.org/record/1188976")
    else:
        train_on_ravdess(data_path)
