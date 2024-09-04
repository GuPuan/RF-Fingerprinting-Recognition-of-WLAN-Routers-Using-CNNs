import copy
import glob
import time
from model import GoogLeNet, Inception
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
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

def train_val_data_process():
    # Find all training data directories
    train_data_dirs = [dir_path for dir_path in glob.glob('model_data/train/*/')]
    print(train_data_dirs)
    train_data = NpyDataset(train_data_dirs)
    num_classes = train_data.get_num_classes()

    # Split the data into training and validation sets (80% train, 20% val)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Create DataLoaders for training and validation datasets
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2)

    # Display dataset statistics
    dataset_statistics(train_data)
    return train_dataloader, val_dataloader, num_classes

def dataset_statistics(dataset):
    # Calculate and print the distribution of labels in the dataset
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f'Label: {label}, Train Data Count: {count}')

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # Set the device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the Adam optimizer with a learning rate of 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Use CrossEntropyLoss as the loss function
    criterion = nn.CrossEntropyLoss()
    # Move the model to the specified device
    model = model.to(device)
    # Copy the current model's parameters
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialize variables to track training progress
    best_acc = 0.0  # Best accuracy
    train_loss_all = []  # List to store training loss values
    val_loss_all = []  # List to store validation loss values
    train_acc_all = []  # List to store training accuracy values
    val_acc_all = []  # List to store validation accuracy values
    since = time.time()  # Start time for training

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print("-"*10)

        # Initialize metrics for the current epoch
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        # Training phase
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # Move inputs and labels to the device
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # Set the model to training mode
            model.train()

            # Forward pass: compute predictions
            output = model(b_x)
            # Get the predicted labels by finding the index of the max log-probability
            pre_lab = torch.argmax(output, dim=1)
            # Compute the loss for the batch
            loss = criterion(output, b_y)

            # Zero the gradients before backpropagation
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Update model parameters
            optimizer.step()

            # Accumulate loss and correct predictions
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # Validation phase
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # Move inputs and labels to the device
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # Set the model to evaluation mode
            model.eval()

            # Forward pass: compute predictions
            output = model(b_x)
            # Get the predicted labels by finding the index of the max log-probability
            pre_lab = torch.argmax(output, dim=1)
            # Compute the loss for the batch
            loss = criterion(output, b_y)

            # Accumulate validation loss and correct predictions
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)

        # Calculate and store loss and accuracy for the epoch
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        print(f"Epoch {epoch} train loss: {train_loss_all[-1]:.4f} train acc: {train_acc_all[-1]:.4f}")
        print(f"Epoch {epoch} val loss: {val_loss_all[-1]:.4f} val acc: {val_acc_all[-1]:.4f}")

        # Update the best model if the current validation accuracy is better
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # Calculate and display time taken for the epoch
        time_use = time.time() - since
        print(f"Time taken for epoch {epoch}: {time_use//60:.0f}m {time_use%60:.0f}s")

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model
    torch.save(best_model_wts, "./best_model.pth")

    # Create a DataFrame to track training and validation statistics
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    })

    return train_process

def matplot_acc_loss(train_process):
    # Create and open a new text file to log results
    log_file = open("b4_9.txt", "w")

    # Plot accuracy and loss over epochs
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
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
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Log loss information to the file
    loss_info = f"Train Losses:\n{train_process.train_loss_all.to_string(index=True)}\n\n"
    loss_info += f"Validation Losses:\n{train_process.val_loss_all.to_string(index=True)}\n\n"
    log_file.write(loss_info)

    # Display the plot
    plt.show()

    # Close the log file
    log_file.close()

if __name__ == '__main__':
    # Process the training and validation data
    train_data, val_data, num_classes = train_val_data_process()

    # Initialize the model with the correct number of classes
    GoogLeNet = GoogLeNet(Inception, num_classes=num_classes)

    # Train the model
    train_process = train_model_process(GoogLeNet, train_data, val_data, num_epochs=50)

    # Plot and log training and validation accuracy/loss
    matplot_acc_loss(train_process)
