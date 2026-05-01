import cv2
from fer.fer import FER

# Start camera and detector
camera = cv2.VideoCapture(0)
detector = FER()

print("Press Q on your keyboard to stop.")

# Keep running until user presses Q
while True:
    # Read current frame from camera
    success, frame = camera.read()
    
    if not success:
        break
    
    # Detect emotion in this frame
    result = detector.detect_emotions(frame)
    
    if result:
        # Get the strongest emotion
        emotions = result[0]['emotions']
        top_emotion = max(emotions, key=emotions.get)
        confidence = emotions[top_emotion]
        
        # Write emotion text on the video
        text = f"{top_emotion}: {confidence:.0%}"
        cv2.putText(frame, text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
    
    # Show the video on screen
    cv2.imshow("Emotion Detector", frame)
    
    # Stop if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
print("Stopped.")
