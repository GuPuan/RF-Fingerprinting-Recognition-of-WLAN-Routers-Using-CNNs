import copy
import glob
import time

from model import LeNet
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
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

# Function to process the training and validation data
def train_val_data_process():
    # Get all directories under the training data folder
    train_data_dirs = [dir_path for dir_path in glob.glob('model_data/train/*/')]
    print(train_data_dirs)
    train_data = NpyDataset(train_data_dirs)  # Create a dataset object
    num_classes = train_data.get_num_classes()  # Get the number of classes

    # Split the data into training and validation sets (80/20 split)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Create data loaders for training and validation
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)
    dataset_statistics(train_data)  # Print dataset statistics
    return train_dataloader, val_dataloader, num_classes

# Function to print dataset statistics
def dataset_statistics(dataset):
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(labels)  # Count the occurrences of each label
    for label, count in label_counts.items():
        print(f'Label: {label}, Train Data Count: {count}')

# Function to train the model
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the Adam optimizer with a learning rate of 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Use CrossEntropyLoss as the loss function
    criterion = nn.CrossEntropyLoss()
    # Move the model to the selected device
    model = model.to(device)
    # Copy the initial model parameters
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize variables to track the best accuracy and losses
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()  # Record the start time

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Initialize variables to track loss and accuracy for this epoch
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # Training phase
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()

            output = model(b_x)  # Forward pass
            pre_lab = torch.argmax(output, dim=1)  # Predict labels
            loss = criterion(output, b_y)  # Calculate loss

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            train_loss += loss.item() * b_x.size(0)  # Accumulate loss
            train_corrects += torch.sum(pre_lab == b_y.data)  # Accumulate correct predictions
            train_num += b_x.size(0)  # Accumulate number of samples

        # Validation phase
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            output = model(b_x)  # Forward pass
            pre_lab = torch.argmax(output, dim=1)  # Predict labels
            loss = criterion(output, b_y)  # Calculate loss

            val_loss += loss.item() * b_x.size(0)  # Accumulate loss
            val_corrects += torch.sum(pre_lab == b_y.data)  # Accumulate correct predictions
            val_num += b_x.size(0)  # Accumulate number of samples

        # Compute and store loss and accuracy for training and validation
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # Update the best model if the current validation accuracy is better
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # Calculate and print the time taken for this epoch
        time_use = time.time() - since
        print("Training and validation time {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model weights
    torch.save(best_model_wts, "./best_model.pth")

    # Create a DataFrame to track the training process
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    })

    return train_process

# Function to plot accuracy and loss
def matplot_acc_loss(train_process):
    # Create and open a new text file to write logs
    log_file = open("b4_20.txt", "w")

    # Plot the training and validation accuracy and loss
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("acc", fontsize=18)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Log accuracy information to the file
    acc_info = f"Train Accuracies:\n{train_process.train_acc_all.to_string(index=True)}\n\n"
    acc_info += f"Validation Accuracies:\n{train_process.val_acc_all.to_string(index=True)}\n\n"
    log_file.write(acc_info)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend(fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Log loss information to the file
    loss_info = f"Train Losses:\n{train_process.train_loss_all.to_string(index=True)}\n\n"
    loss_info += f"Validation Losses:\n{train_process.val_loss_all.to_string(index=True)}\n\n"
    log_file.write(loss_info)

    # Show the plot
    plt.show()

    # Close the log file
    log_file.close()

# Main script execution
if __name__ == '__main__':
    # Process training and validation data
    train_data, val_data, num_classes = train_val_data_process()
    # Initialize the model with the number of classes
    LeNet = LeNet(num_classes=num_classes)
    # Train the model
    train_process = train_model_process(LeNet, train_data, val_data, num_epochs=50)
    # Plot the training results
    matplot_acc_loss(train_process)
