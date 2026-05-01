import cv2

# This opens your webcam
# 0 means "first camera" (your built-in webcam)
camera = cv2.VideoCapture(0)

# Read one frame (one photo) from the camera
success, photo = camera.read()

# If camera worked successfully
if success:
    # Save the photo as a file
    cv2.imwrite("my_face.jpg", photo)
    print("✅ Photo saved! Check your folder.")
else:
    print("❌ Camera not found. Check if webcam is connected.")

# Always close the camera when done
camera.release()
