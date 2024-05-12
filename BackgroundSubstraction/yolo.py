from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "output2.avi"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])
passed_ids = []
passed_forward = 0
passed_inward = 0

line_height = 540

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1920,1080))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[2, 5, 7])

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) == 1:
                continue
            if track_id not in passed_ids and line_height - 10 < y < line_height + 10:
                passed_ids.append(track_id)
                if track[-1][1] - track[-2][1] < 0:  # passed forward
                    passed_forward += 1
                else:  # passed inward
                    passed_inward += 1
            if len(track) > 30:  # limiting tracking
                track.pop(0)

        # Display the annotated frame
        cv2.line(annotated_frame, [0, 540], [1920, 540], color=(21, 50, 80), thickness=10)
        cv2.putText(annotated_frame, f"Forward: {passed_forward}; Inward: {passed_inward}", (15, 1080 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    3)
        out.write(annotated_frame)  # saving result

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()