import os
import re

def extract_unique_labels(folder_path, output_file):
    unique_labels = set()
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ensure it's a text file
        if os.path.isfile(file_path) and filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    match = re.search(r":\s*(\S+)$", line.strip())  # Extract label
                    if match:
                        unique_labels.add(match.group(1))
    
    # Sort labels and create mapping
    sorted_labels = sorted(unique_labels)
    
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, label in enumerate(sorted_labels):
            out.write(f"{idx} {label}\n")
    
    print(f"Mapping file saved to {output_file}")

# Example usage
folder_path = "data/production/groundTruth"  # Change this to your folder path
output_file = "data/production/mapping.txt"
extract_unique_labels(folder_path, output_file)