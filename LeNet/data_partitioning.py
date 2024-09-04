import os
import random
from shutil import copy, rmtree

def mkfile(file):
    # Create a directory if it doesn't exist
    if not os.path.exists(file):
        os.makedirs(file)

# Define the root directory of the data and the output directory
root_path = 'initial_data/extracted_data'
output_path = 'model_data'

# If the model_data directory exists, remove it
if os.path.exists(output_path):
    rmtree(output_path)

# Recreate the model_data directory
mkfile(output_path)

# Get the names of all folders in the root directory
data_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

# Create train and test folders under model_data
mkfile(os.path.join(output_path, 'train'))
mkfile(os.path.join(output_path, 'test'))

# Iterate over each data folder
for folder in data_folders:
    folder_path = os.path.join(root_path, folder)

    # Create corresponding folders under model_data/train and model_data/test
    mkfile(os.path.join(output_path, 'train', folder))
    mkfile(os.path.join(output_path, 'test', folder))

    # Get all files in the current data folder
    files = os.listdir(folder_path)
    num_files = len(files)
    split_rate = 0.1  # Define the proportion of files to be used for the test set
    eval_index = random.sample(files, k=int(num_files * split_rate))  # Randomly select files for the test set

    # Distribute files to train and test folders based on the split rate
    for index, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        if file in eval_index:
            new_path = os.path.join(output_path, 'test', folder)  # Copy to the test folder
        else:
            new_path = os.path.join(output_path, 'train', folder)  # Copy to the train folder
        copy(file_path, new_path)  # Copy the file to the new location
        print("\r[{}] processing [{}/{}]".format(folder, index + 1, num_files), end="")  # Display processing progress
    print()  # Move to the next line after processing each folder

print("Processing done!")  # Indicate that the processing is complete
