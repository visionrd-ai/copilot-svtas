import os
import subprocess

# Set the folder path
folder_path = "data/thal/Videos"  # Change this to your folder path

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".mp4"):
        input_path = os.path.join(folder_path, filename)
        temp_output_path = os.path.join(folder_path, "temp_" + filename)

        # FFmpeg command to re-encode as MJPEG
        command = [
            "ffmpeg", "-y", "-i", input_path, "-c:v", "mjpeg", "-q:v", "2", "-an", temp_output_path
        ]

        # Run FFmpeg
        subprocess.run(command, check=True)

        # Overwrite the original file
        os.replace(temp_output_path, input_path)
        print(f"Processed and overwritten: {filename}")

print("Processing complete.")
