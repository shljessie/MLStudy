import os
import shutil

def replicate_files(directory):
    # Define the target directory for replicated files
    replicated_dir = os.path.join(directory, "replicated")
    os.makedirs(replicated_dir, exist_ok=True)
    
    # Track replicated files for confirmation message
    replicated_files = []
    
    # Iterate through each file in the given directory
    for filename in os.listdir(directory):
        # Only process files, not directories
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Define the replicated file name and path
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_replicated{ext}"
            new_file_path = os.path.join(replicated_dir, new_filename)
            
            # Replicate file content line by line
            try:
                with open(file_path, 'r') as original_file, open(new_file_path, 'w') as new_file:
                    for line in original_file:
                        new_file.write(line)
                replicated_files.append(new_filename)
            except IOError as e:
                print(f"Error processing {filename}: {e}")
    
    # Confirmation message
    if replicated_files:
        print("Replication successful! Created the following files:")
        for file in replicated_files:
            print(file)
    else:
        print("No files were replicated.")

# Example usage
replicate_files('/path/to/your/directory')
