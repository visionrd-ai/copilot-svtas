import os
import subprocess

# Input paths
video_dir = 'data/thal/TEST_VID'  
label_dir = 'data/thal/TEST_GT' 
output_dir = 'data/thal/TEST_TRIMMED'
os.makedirs(output_dir, exist_ok=True)

# Iterate over all videos in the video directory
for video_file in os.listdir(video_dir):
    if not video_file.lower().endswith('.mp4'):
        continue
    video_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_dir, video_file)
    label_path = os.path.join(label_dir, f'{video_name}.txt')

    if not os.path.exists(label_path):
        print(f'No label file found for {video_name}, skipping...')
        continue

    # Read labels from the file
    with open(label_path, 'r') as f:
        labels = [line.strip().split(': ') for line in f.readlines()]
        labels = [(int(frame), label) for frame, label in labels]

    # Extract segments of non-background labels
    segments = []
    current_label = None
    start_frame = None

    for frame, label in labels:
        if label != 'background':
            if label != current_label:
                if current_label is not None:
                    segments.append((start_frame, frame - 1, current_label))
                start_frame = frame
                current_label = label
        else:
            if current_label is not None:
                segments.append((start_frame, frame - 1, current_label))
                current_label = None

    if current_label is not None:
        segments.append((start_frame, frame, current_label))

    # Create a directory for each video
    video_output_dir = os.path.join(output_dir, 'Videos')
    os.makedirs(video_output_dir, exist_ok=True)

    label_output_dir = os.path.join(output_dir, 'groundTruth')
    os.makedirs(label_output_dir, exist_ok=True)

    # Trim videos and create new label files
    fps = 30  # Assuming 30 frames per second
    for start_frame, end_frame, label in segments:
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        output_file = os.path.join(video_output_dir, f'{video_name}_{label}_{start_frame}_{end_frame}.mp4')
        label_file = os.path.join(label_output_dir, f'{video_name}_{label}_{start_frame}_{end_frame}.txt')

        # Trim the video
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start_time), '-t', str(duration),
            '-c:v', 'copy', output_file
        ]
        subprocess.run(command)
        print(f'Created {output_file}')

        with open(label_file, 'w') as lf:
            for frame, lbl in labels:
                if start_frame <= frame <= end_frame:
                    lf.write(f'{frame - start_frame}: {lbl}\n')
        print(f'Created {label_file}')

print('Video trimming and label clipping completed!')
