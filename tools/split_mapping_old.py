import re

def create_mapping(input_file, output_file):
    branches = set()
    actions = set()
    
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ", 1)  # Split index and label
            if len(parts) == 2:
                label = parts[1]
                if label == "bg":
                    actions.add("bg")
                else:
                    split_label = label.split("_")
                    if len(split_label) > 1:
                        branches.add(split_label[0])
                        actions.add("_".join(split_label[1:]))
    
    # Combine and sort all unique labels
    sorted_labels = sorted(branches) + sorted(actions)
    
    # Write mapping file
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, label in enumerate(sorted_labels):
            out.write(f"{idx} {label}\n")
    
    print(f"Mapping file saved to {output_file}")

# Example usage
input_file = "data/production/mapping.txt"  # Change this to your input file path
output_file = "data/production/mapping_split.txt"
create_mapping(input_file, output_file)