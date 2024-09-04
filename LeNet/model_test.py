import glob

import torch.utils.data as Data
from model import LeNet
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Custom Dataset class for loading .npy files
class NpyDataset(Dataset):
    def __init__(self, data_dirs):
        self.data = []
        self.labels = []
        for data_dir in data_dirs:
            # Load all .npy files from the directory
            npy_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')]
            for file in npy_files:
                npy_data = np.load(file)  # Load the .npy file
                self.data.append(npy_data)
                # Adjust label indexing to extract the label from the directory name
                label = int(os.path.basename(os.path.dirname(data_dir)).split('_')[0])
                self.labels.append(label)
        # Convert data and labels to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.num_classes = len(set(self.labels))  # Get the number of unique classes

    def __len__(self):
        return len(self.data)  # Return the total number of samples

    def __getitem__(self, idx):
        data = self.data[idx]
        data = np.expand_dims(data, axis=0)  # Add a channel dimension
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def get_num_classes(self):
        return self.num_classes  # Return the number of classes

# Function to process the test data
def test_data_process():
    # Get all directories under the test data folder
    test_data_dirs = [dir_path for dir_path in glob.glob('model_data/test/*/')]
    print(test_data_dirs)
    test_data = NpyDataset(test_data_dirs)  # Create a dataset object
    num_classes = test_data.get_num_classes()  # Get the number of classes

    dataset_statistics(test_data)  # Print dataset statistics

    # Create a subset of the test data
    test_size = len(test_data)
    test_data_subset = torch.utils.data.Subset(test_data, range(test_size))

    # Create a data loader for the test data
    test_dataloader = DataLoader(test_data_subset, batch_size=32, shuffle=True, num_workers=2)

    return test_dataloader, num_classes

# Function to print dataset statistics
def dataset_statistics(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(labels)  # Count the occurrences of each label
    for label, count in label_counts.items():
        print(f'Label: {label}, Train Data Count: {count}')

# Function to test the model
def test_model_process(model, test_dataloader):
    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Move the model to the selected device
    model = model.to(device)

    # Initialize parameters for tracking accuracy
    test_corrects = 0.0
    test_num = 0

    # Perform the testing process without computing gradients to save memory and speed up
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # Move the data to the selected device
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            model.eval()  # Set the model to evaluation mode
            output = model(test_data_x)  # Forward pass
            pre_lab = torch.argmax(output, dim=1)  # Get the predicted labels
            test_corrects += torch.sum(pre_lab == test_data_y.data)  # Accumulate correct predictions
            test_num += test_data_x.size(0)  # Accumulate the number of samples

    # Calculate and print the test accuracy
    test_acc = test_corrects.double().item() / test_num
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    # Load the test data and model
    test_dataloader, num_classes = test_data_process()

    # Initialize the model with the number of classes
    model = LeNet(num_classes=num_classes)
    model.load_state_dict(torch.load('best_model.pth'))  # Load the best model weights
    # Test the model
    test_model_process(model, test_dataloader)

    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
