import cv2
import numpy as np
from sort import Sort

# Create the car classifier (or any single object classifier)
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for the video file
cap = cv2.VideoCapture('cars.avi')

if not cap.isOpened():
    print("Error opening video stream or file")

# Get the original frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Create an instance of SORT with potentially adjusted parameters for single object tracking
mot_tracker = Sort(max_age=10, min_hits=3)  # Adjust these values based on your needs

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=4)

    detections = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
    if detections.size == 0:
        detections = np.empty((0, 4))

    track_bbs_ids = mot_tracker.update(detections)

    # Visualization: Draw bounding box and track ID for a single object
    if len(track_bbs_ids) > 0:
        x1, y1, x2, y2, track_id = track_bbs_ids[0]  # Focus on the first track if multiple are present
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
        cv2.putText(frame, str(int(track_id)), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

