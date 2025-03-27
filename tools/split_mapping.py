import re

def split_mapping(mapping_file, output_branches, output_actions):
    branches = set()
    actions = set()
    
    with open(mapping_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ", 1)  # Split index and label
            if len(parts) == 2:
                label = parts[1]
                if label == "bg":
                    branches.add("bg")
                    actions.add("bg")
                else:
                    split_label = label.split("_")
                    if len(split_label) > 1:
                        branches.add(split_label[0])
                        actions.add("_".join(split_label[1:]))
    
    # Sort and write branch mappings
    sorted_branches = sorted(branches)
    with open(output_branches, "w", encoding="utf-8") as out:
        for idx, branch in enumerate(sorted_branches):
            out.write(f"{idx} {branch}\n")
    
    # Sort and write action mappings
    sorted_actions = sorted(actions)
    with open(output_actions, "w", encoding="utf-8") as out:
        for idx, action in enumerate(sorted_actions):
            out.write(f"{idx} {action}\n")
    
    print(f"Branch mapping file saved to {output_branches}")
    print(f"Action mapping file saved to {output_actions}")

# Example usage
mapping_file = "data/production/mapping.txt"  # Change this to your mapping file path
output_branches = "data/production/mapping_branches.txt"
output_actions = "data/production/mapping_actions.txt"
split_mapping(mapping_file, output_branches, output_actions)
