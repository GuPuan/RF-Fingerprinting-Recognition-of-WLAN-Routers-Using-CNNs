import glob
import torch.utils.data as Data
from model import GoogLeNet, Inception
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class NpyDataset(Dataset):
    def __init__(self, data_dirs):
        self.data = []
        self.labels = []
        # Load .npy files from the given directories
        for data_dir in data_dirs:
            npy_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.npy')]
            for file in npy_files:
                npy_data = np.load(file)
                self.data.append(npy_data)
                # Extract label from the directory name, assuming it's in the format 'label_xx'
                label = int(os.path.basename(os.path.dirname(data_dir)).split('_')[0])
                self.labels.append(label)
        # Convert data and labels to numpy arrays
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        # Determine the number of classes
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        # Expand dimensions to include channel information
        data = np.expand_dims(data, axis=0)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

    def get_num_classes(self):
        return self.num_classes  # Return the number of classes

def test_data_process():
    # Find all test data directories
    test_data_dirs = [dir_path for dir_path in glob.glob('model_data/test/*/')]
    print(test_data_dirs)
    test_data = NpyDataset(test_data_dirs)
    num_classes = test_data.get_num_classes()

    # Display statistics about the dataset
    dataset_statistics(test_data)

    # Prepare the test dataset and dataloader
    test_size = len(test_data)
    test_data_subset = torch.utils.data.Subset(test_data, range(test_size))

    # Create a DataLoader for the test dataset
    test_dataloader = DataLoader(test_data_subset, batch_size=32, shuffle=True, num_workers=2)

    return test_dataloader, num_classes

def dataset_statistics(dataset):
    # Calculate and print the distribution of labels in the dataset
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f'Label: {label}, Train Data Count: {count}')

def test_model_process(model, test_dataloader):
    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Move the model to the specified device
    model = model.to(device)

    # Initialize parameters for tracking accuracy
    test_corrects = 0.0
    test_num = 0

    # Disable gradient computation to save memory and improve speed during inference
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # Move data to the specified device
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            # Set the model to evaluation mode
            model.eval()
            # Forward pass: get predictions for the test data
            output = model(test_data_x)
            # Get the predicted labels by finding the index of the max log-probability
            pre_lab = torch.argmax(output, dim=1)
            # Update the correct predictions count
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # Update the total number of test samples
            test_num += test_data_x.size(0)

    # Calculate and print the test accuracy
    test_acc = test_corrects.double().item() / test_num
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    # Load the test data and determine the number of classes
    test_dataloader, num_classes = test_data_process()

    # Initialize the model with the correct number of classes
    model = GoogLeNet(Inception, num_classes=num_classes)
    model.load_state_dict(torch.load('best_model.pth'))

    # Test the model using the test dataset
    test_model_process(model, test_dataloader)

    # Set the device to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
