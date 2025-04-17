import os

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

# Example usage
folder_path = "data/thal/groundTruth"  # Change this to your folder path
prefixes = ["2"]  # Add your prefixes here
test_bundle = "data/thal/splits/test.bundle"
train_bundle = "data/thal/splits/train.bundle"

split_files_by_prefix(folder_path, prefixes, test_bundle, train_bundle)