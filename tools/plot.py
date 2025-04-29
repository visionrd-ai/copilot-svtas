import cv2
import os 

def plot_labels_on_video(input_video_path, labels_txt_path1, labels_txt_path2, output_video_path, font_scale=1, font_thickness=2):
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Read labels from the text files
    with open(labels_txt_path1, 'r') as f1, open(labels_txt_path2, 'r') as f2:
        labels1 = f1.read().splitlines()
        labels2 = f2.read().splitlines()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Clip label lengths to min(label count, frame count)
    max_frames = min(len(labels1), len(labels2), total_frames)

    if len(labels1) != len(labels2) or max_frames != total_frames:
        print(f"Warning: Mismatched lengths - using {max_frames} frames.")

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position1 = (50, 50)
    text_position2 = (50, 100)

    frame_index = 0
    while frame_index < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        label1 = 'Pred: ' + labels1[frame_index]
        label2 = 'GT: ' + labels2[frame_index]

        cv2.putText(frame, label1, text_position1, font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
        cv2.putText(frame, label2, text_position2, font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Plotted video saved at {output_video_path}")

# Example usage
input_video = 'data/dep_outs/2_480p_recordings.mp4'
labels_txt1 = 'data/dep_outs/2_preds.txt'
labels_txt2 = 'data/dep_outs/2_preds.txt'
output_video = 'vis/2_deployed.mp4'
os.makedirs('vis', exist_ok=True)
plot_labels_on_video(input_video, labels_txt1, labels_txt2, output_video)
