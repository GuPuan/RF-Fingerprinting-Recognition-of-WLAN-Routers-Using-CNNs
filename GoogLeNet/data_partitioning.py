import os
import random
from shutil import copy, rmtree

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# Define the root directory and output directory
root_path = 'initial_data/extracted_data'
output_path = 'model_data'

# If the model_data folder exists, delete it
if os.path.exists(output_path):
    rmtree(output_path)

# Recreate the model_data directory
mkfile(output_path)

# Get all folder names in the root directory
data_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

# Create train and test folders
mkfile(os.path.join(output_path, 'train'))
mkfile(os.path.join(output_path, 'test'))

# Iterate through each data folder
for folder in data_folders:
    folder_path = os.path.join(root_path, folder)

    # Create corresponding folders in model_data/train and model_data/test
    mkfile(os.path.join(output_path, 'train', folder))
    mkfile(os.path.join(output_path, 'test', folder))

    # Get all files in the current folder
    files = os.listdir(folder_path)
    num_files = len(files)
    split_rate = 0.1
    eval_index = random.sample(files, k=int(num_files * split_rate))

    # Distribute files into train and test folders based on the split rate
    for index, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        if file in eval_index:
            new_path = os.path.join(output_path, 'test', folder)
        else:
            new_path = os.path.join(output_path, 'train', folder)
        copy(file_path, new_path)
        print("\r[{}] processing [{}/{}]".format(folder, index + 1, num_files), end="")  # Processing bar
    print()

print("Processing done!")
