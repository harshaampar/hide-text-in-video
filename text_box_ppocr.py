import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm
import sys

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='en', det=True, rec=False)

# Load video
try:
    video_path = sys.argv[1]  # Replace with your video file path
except:
    print(f'Usage : python {__file__} input_video output_video(optional)')
    exit(-1)
    
cap = cv2.VideoCapture(video_path)

# Set up video writer
try:
    output_path = sys.argv[2]
except:
    output_path = 'output_video_with_black_boxes.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Couldn't read the video")

# Manually select the card region in the first frame
bbox = cv2.selectROI("Select Card", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Card")

# Initialize tracker with the first frame and the selected bounding box
tracker = cv2.TrackerCSRT_create()  # You can also try cv2.TrackerKCF_create() for faster tracking
tracker.init(first_frame, bbox)

# Get total frames for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Process video
with tqdm(total=total_frames, desc="Processing Video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker and get the new position of the card
        success, bbox = tracker.update(frame)
        if success:
            # Extract the current card region
            x, y, w, h = [int(v) for v in bbox]
            card_frame = frame[y:y+h, x:x+w]

            # Detect text in the card region using PaddleOCR
            # result = ocr.ocr(card_frame, cls=False)
            if card_frame.size != 0:
                result = ocr.ocr(card_frame, rec=False, cls=False)
                # Black out detected text in the card region

                try:
                    if result is not None and len(result) > 0:
                        for line in result[0]:
                            points = np.array(line, dtype=np.int32)
                            _, _, ww, hh = cv2.boundingRect(points)
                            # Adjust points for the position within the original frame
                            points[:, 0] += x
                            points[:, 1] += y
                            # Draw filled black polygon over text area
                            if hh < 30:
                                cv2.fillPoly(frame, [points], color=(0, 0, 0))
                except Exception as e:
                    print(e)

        # Write the frame to the output video
        out.write(frame)

        pbar.update(1)
        cv2.imshow('Video with Black Box', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
