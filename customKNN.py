import cv2
import numpy as np
from collections import deque

frame_size = 10
threshold = 30
k = 3

background_frames = deque(maxlen=frame_size)

def knn_background_subtraction(frame, background_frames, k, threshold):
    frame = frame.astype(np.int16)
    background_stack = np.stack(background_frames, axis=0)
    differences = np.abs(background_stack - frame)
    sorted_differences = np.sort(differences, axis=0)
    min_k_distances = sorted_differences[:k]
    max_distance = np.max(min_k_distances, axis=0)
    mask = (max_distance > threshold).astype(np.uint8) * 255
    return mask

cap = cv2.VideoCapture("video.mp4")

# VideoWriter configuration
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height), isColor=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(background_frames) < frame_size:
        background_frames.append(gray)
    else:
        mask = knn_background_subtraction(gray, background_frames, k, threshold)
        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)

        # Write the mask to the output video
        out.write(mask)

    background_frames.append(gray)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
