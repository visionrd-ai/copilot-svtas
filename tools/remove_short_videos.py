import os

# Define paths
directory = "data/production/groundTruth"  # Change this to the directory containing the files
filenames_txt = "data/production/splits/test.bundle"  # Change this to the path of the txt file containing filenames

# Read the list of filenames
with open(filenames_txt, "r") as f:
    filenames = [line.strip() for line in f.readlines()]

# Filter out files with less than 32 lines
valid_filenames = []
for filename in filenames:
    file_path = os.path.join(directory, filename)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            line_count = sum(1 for _ in f)
        if line_count >= 32:
            valid_filenames.append(filename)

# Overwrite the original txt file with valid filenames
with open(filenames_txt, "w") as f:
    for filename in valid_filenames:
        f.write(filename + "\n")

print("Filtering complete. Updated file list saved.")
