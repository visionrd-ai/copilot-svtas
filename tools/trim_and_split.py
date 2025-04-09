import os
import subprocess
import re

# ------------------- Trimming Part -------------------

video_dir = 'data/thal/Videos'  
label_dir = 'data/thal/groundTruth' 
output_dir = 'data/thal/'
os.makedirs(output_dir, exist_ok=True)

min_segment_length = 32  # Minimum number of frames per segment
fps = 30  # Assuming 30 frames per second

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
                    if frame - start_frame >= min_segment_length:
                        segments.append((start_frame, frame - 1, current_label))
                start_frame = frame
                current_label = label
        else:
            if current_label is not None:
                if frame - start_frame >= min_segment_length:
                    segments.append((start_frame, frame - 1, current_label))
                current_label = None

    if current_label is not None and (labels[-1][0] - start_frame + 1) >= min_segment_length:
        segments.append((start_frame, labels[-1][0], current_label))

    # Create output directories
    video_output_dir = os.path.join(output_dir, 'Videos')
    os.makedirs(video_output_dir, exist_ok=True)

    label_output_dir = os.path.join(output_dir, 'groundTruth')
    os.makedirs(label_output_dir, exist_ok=True)

    # Trim videos and create new label files
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

# ------------------- Splitting Part -------------------

def split_files_by_prefix(folder_path, prefixes, test_bundle, train_bundle):
    with open(test_bundle, "w", encoding="utf-8") as test_file, \
         open(train_bundle, "w", encoding="utf-8") as train_file:
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isfile(file_path):
                if any(filename.startswith(prefix) for prefix in prefixes):
                    test_file.write(filename + "\n")
                else:
                    train_file.write(filename + "\n")
    
    print(f"Test bundle saved to {test_bundle}")
    print(f"Train bundle saved to {train_bundle}")

# Create splits after trimming
split_folder = os.path.join(output_dir, "groundTruth")
prefixes = ["8"]  # Add desired prefixes here
splits_dir = os.path.join(output_dir, "splits")
os.makedirs(splits_dir, exist_ok=True)

test_bundle = os.path.join(splits_dir, "test.bundle")
train_bundle = os.path.join(splits_dir, "train.bundle")

split_files_by_prefix(split_folder, prefixes, test_bundle, train_bundle)

# ------------------- Label Post-Processing Part -------------------

def process_label_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)

            base_name = os.path.splitext(filename)[0]
            output_b_number_path = os.path.join(output_folder, f"{base_name}_branch.txt")
            output_word_path = os.path.join(output_folder, f"{base_name}.txt")
          
            with open(input_path, "r") as infile, \
                 open(output_b_number_path, "w") as b_number_file, \
                 open(output_word_path, "w") as word_file:
                for line in infile:
                    parts = line.strip().split(": ", 1)
                    if len(parts) == 2:
                        label = parts[1]

                        if label == "bg":
                            b_number_file.write("bg\n")
                            word_file.write("bg\n")
                        else:
                            match = re.match(r'(B\d+)_([a-zA-Z]+)', label)
                            if match:
                                b_number = match.group(1)
                                word = match.group(2)
                                b_number_file.write(f"{b_number}\n")
                                word_file.write(f"{word}\n")

    print(f"Processing complete. Output files saved in {output_folder}")

# Run label post-processing
input_folder = os.path.join(output_dir, "groundTruth")
output_folder = os.path.join(output_dir, "groundTruth_split")

process_label_files(input_folder, output_folder)
