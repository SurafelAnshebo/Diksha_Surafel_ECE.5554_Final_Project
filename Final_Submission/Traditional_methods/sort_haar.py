import cv2
import numpy as np
from sort import Sort  # Make sure SORT is correctly installed

# Create our car classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('cars.avi')

# Initialize SORT tracker
tracker = Sort()

# Loop once video is successfully loaded
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars using Haar cascades
    cars = car_classifier.detectMultiScale(frame_gray, 1.4, 2)
    
    # Format detections for SORT (convert from x, y, w, h to x1, y1, x2, y2)
    dets = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
    print((dets))
    
    # If detections were made, update SORT tracker with detections
    if len(dets) > 0:
        tracked_objects = tracker.update(dets)
        # Draw tracking boxes
        for *coords, _ in tracked_objects:
            x1, y1, x2, y2 = map(int, coords)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        # Even with no detections, we still need to call update to allow for trackers to predict
        tracker.update(np.empty((0, 5)))

    cv2.imshow('Cars', frame)

    if cv2.waitKey(1) == 13:
        break
      # 13 is the Enter Key
        
cap.release()
cv2.destroyAllWindows()