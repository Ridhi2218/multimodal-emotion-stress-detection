from fer.fer import FER
import cv2

# Load the photo we took in step 1
my_photo = cv2.imread("my_face.jpg")

# Create an emotion detector
# This is the CNN (AI model) your paper talks about
detector = FER()

# Ask the AI to detect emotions in the photo
result = detector.detect_emotions(my_photo)

# Print the result
if result:
    emotions = result[0]['emotions']
    print("Here are the emotions detected:")
    print("")
    for emotion, score in emotions.items():
        bar = "█" * int(score * 20)  # Visual bar
        print(f"{emotion:10} {bar} {score:.2f}")
else:
    print("No face detected in photo.")
