# author pattadon
import cv2
import os
from datetime import datetime

# create folder image (if don't exist)
output_dir = "captured_faces"
os.makedirs(output_dir, exist_ok=True)

# load Haar Cascade detected faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Open the default camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("can't open camera")
    exit()

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("can't read frame")
        break

    # tranfer black white (Grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # detect faces 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # draw rectangle and take automatically pictures
    for (x, y, w, h) in faces:
        # draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # # save face image
        face_img = frame[y:y + h, x:x + w]  # crop face only
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # define name file follow time
        filename = os.path.join(output_dir, f"face_{timestamp}.jpg")
        cv2.imwrite(filename, face_img)  # save image face


    # Write the frame to the output file
    out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()