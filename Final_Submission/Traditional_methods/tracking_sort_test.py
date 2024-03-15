import cv2
import numpy as np
from sort import Sort

# Create the car classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for the video file
cap = cv2.VideoCapture('cars_.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Get the original frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
# Calculate the delay in milliseconds between frames
delay = int(1000 / fps)

# Create an instance of SORT
mot_tracker = Sort()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars using the Haar cascade
    cars = car_classifier.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=4)
    #Check this out --- I play with these numbers 
    
    # Format detections for SORT
    # Ensure detections is always a 2D array, even when no cars are detected
    detections = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
    if detections.size == 0:
        detections = np.empty((0, 4))  # Adjust shape according to your SORT implementation's needs

    # Update SORT with detections
    track_bbs_ids = mot_tracker.update(detections)	# I think this part is for assigning ID

    # Visualization: Draw bounding boxes and track IDs
    for j, (x1, y1, x2, y2, track_id) in enumerate(track_bbs_ids):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.putText(frame, str(int(track_id)), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Cars', frame)

    # Break the loop when 'q' is pressed, using the calculated delay to match video's original frame rate
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()

