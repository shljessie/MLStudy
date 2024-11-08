import os
import shutil

def replicate_files_recursive(directory):
    # Define the target directory for replicated files
    replicated_dir = os.path.join(directory, "replicated")
    os.makedirs(replicated_dir, exist_ok=True)
    
    # Track replicated files for confirmation message
    replicated_files = []
    
    # Walk through all files and subdirectories in the given directory
    for root, _, files in os.walk(directory):
        # Define the corresponding replicated subdirectory path
        relative_path = os.path.relpath(root, directory)
        target_subdir = os.path.join(replicated_dir, relative_path)
        os.makedirs(target_subdir, exist_ok=True)
        
        for filename in files:
            # Skip files in the "replicated" folder to avoid re-replication
            if "replicated" in os.path.relpath(root, directory).split(os.sep):
                continue
            
            # Define the original and replicated file paths
            original_file_path = os.path.join(root, filename)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_replicated{ext}"
            new_file_path = os.path.join(target_subdir, new_filename)
            
            # Replicate file content line by line
            try:
                with open(original_file_path, 'r') as original_file, open(new_file_path, 'w') as new_file:
                    for line in original_file:
                        new_file.write(line)
                replicated_files.append(new_file_path)
            except IOError as e:
                print(f"Error processing {original_file_path}: {e}")
    
    # Confirmation message
    if replicated_files:
        print("Replication successful! Created the following files:")
        for file in replicated_files:
            print(file)
    else:
        print("No files were replicated.")

# Example usage
replicate_files_recursive('/path/to/your/directory')
