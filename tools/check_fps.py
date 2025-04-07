import cv2
import time

# Paths to input and output video files
video_path = "data/Videos/1.mp4"
output_video_path = "data/Videos/1_fixed_fps.mp4"

# Open the input video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get the video's FPS and frame size
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object to save the output video with the same resolution and desired FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables for frame timing
prev_time = time.time()
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure the frame is shown at the right interval based on FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    time_to_wait = (1.0 / fps) - elapsed_time  # Calculate time to wait before next frame

    if time_to_wait > 0:
        time.sleep(time_to_wait)  # Sleep to maintain fixed frame rate
    print("Waiting for ", time_to_wait, "seconds")
    # Write the frame to the output video
    out.write(frame)

    # Optionally, you can print the timestamp or FPS info
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    # print(f"Frame {frame_idx}: {fps} FPS, Timestamp: {timestamp}")

    prev_time = time.time()
    frame_idx += 1

# Release video resources
cap.release()
out.release()

print(f"Video saved to {output_video_path}")
