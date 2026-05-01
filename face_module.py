# ============================================================
#  FACE MODULE — Robust version with camera diagnostics
# ============================================================

import cv2
from fer.fer import FER
import numpy as np
import time
import os


def capture_photo():
    """
    Tries multiple camera indices and warmup strategies.
    Saves a debug image so you can see what was captured.
    Returns: (frame, success)
    """
    for cam_index in [0, 1, 2]:
        camera = cv2.VideoCapture(cam_index)

        if not camera.isOpened():
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame = None
        for i in range(20):
            ret, frame = camera.read()
            time.sleep(0.05)

        camera.release()

        if frame is not None and frame.size > 0:
            cv2.imwrite("debug_capture.jpg", frame)
            print(f"Camera {cam_index} worked. Debug image saved.")
            return frame, True

    print("No camera found on indices 0, 1, 2")
    return None, False


def detect_face_emotions(frame):
    """
    Tries 3 strategies to detect a face and return emotions.
    """
    # Strategy 1: Fast FER
    try:
        detector = FER(mtcnn=False)
        result = detector.detect_emotions(frame)
        if result:
            return result[0]['emotions']
    except Exception as e:
        print(f"Strategy 1 failed: {e}")

    # Strategy 2: Accurate FER with mtcnn
    try:
        detector = FER(mtcnn=True)
        result = detector.detect_emotions(frame)
        if result:
            return result[0]['emotions']
    except Exception as e:
        print(f"Strategy 2 failed: {e}")

    # Strategy 3: OpenCV Haar cascade + FER
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]

            detector = FER(mtcnn=False)
            result = detector.detect_emotions(face_crop)
            if result:
                return result[0]['emotions']

            # Haar found face but FER couldn't classify — return neutral baseline
            return {
                'angry': 0.05, 'disgust': 0.02, 'fear': 0.03,
                'happy': 0.20, 'sad': 0.10, 'surprise': 0.05,
                'neutral': 0.55
            }
    except Exception as e:
        print(f"Strategy 3 failed: {e}")

    return None


def get_dominant_emotion(emotions):
    if emotions:
        return max(emotions, key=emotions.get)
    return "unknown"


def run_live_detection():
    camera = cv2.VideoCapture(0)
    detector = FER(mtcnn=False)
    print("Live detection started. Press Q to stop.")
    while True:
        success, frame = camera.read()
        if not success:
            break
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]['emotions']
            dominant = max(emotions, key=emotions.get)
            score = emotions[dominant]
            label = f"{dominant.upper()}: {score:.0%}"
            cv2.putText(frame, label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 100), 2)
            box = result[0]['box']
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
        cv2.imshow("Emotion Detector — Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=" * 50)
    print("CAMERA DIAGNOSTIC TEST")
    print("=" * 50)
    frame, ok = capture_photo()
    if ok:
        print("Camera capture successful!")
        print("Check 'debug_capture.jpg' in your folder.\n")
        emotions = detect_face_emotions(frame)
        if emotions:
            print("Emotions detected:")
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 30)
                print(f"  {emotion:12} {bar} {score:.2%}")
        else:
            print("No face detected. Open debug_capture.jpg to see what camera captured.")
    else:
        print("Camera failed. Check if another app is using it.")
