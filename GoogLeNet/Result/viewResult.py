import matplotlib.pyplot as plt
import os

# Print the current working directory
print(os.getcwd())

# File path to the data file
file_path = 'n/b0_1.txt'

# Read data from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists to store extracted data
train_accuracies = []
validation_accuracies = []
train_losses = []
validation_losses = []

# Parse the data and store it in the corresponding lists
section = None
for line in lines:
    line = line.strip()
    if not line:
        continue  # Skip empty lines
    if 'Train Accuracies' in line:
        section = 'train_acc'
        continue
    elif 'Validation Accuracies' in line:
        section = 'val_acc'
        continue
    elif 'Train Losses' in line:
        section = 'train_loss'
        continue
    elif 'Validation Losses' in line:
        section = 'val_loss'
        continue

    try:
        value = float(line.split()[1])  # Try to extract and convert the second element to a float
    except (IndexError, ValueError):
        continue  # Skip the line if parsing fails

    # Append the extracted value to the corresponding list based on the current section
    if section == 'train_acc':
        train_accuracies.append(value)
    elif section == 'val_acc':
        validation_accuracies.append(value)
    elif section == 'train_loss':
        train_losses.append(value)
    elif section == 'val_loss':
        validation_losses.append(value)

# Ensure all lists have the same length by trimming them to the shortest length
min_length = min(len(train_accuracies), len(validation_accuracies), len(train_losses), len(validation_losses))
train_accuracies = train_accuracies[:min_length]
validation_accuracies = validation_accuracies[:min_length]
train_losses = train_losses[:min_length]
validation_losses = validation_losses[:min_length]

# Create an array of epochs
epochs = range(1, min_length + 1)

# Set up the figure size and layout
plt.figure(figsize=(6, 6))  # Adjust to a narrower figure

# Font size settings
label_fontsize = 24
legend_fontsize = 22
ticks_fontsize = 24

# Adjust the spacing between subplots
plt.subplots_adjust(left=0.2, right=0.4, top=0.8, bottom=0.48)  # Reduce left and right margins

# Plot loss values for training and validation
plt.plot(epochs, train_losses, 'bo-', label='Train Loss')  # Blue dots with line for training loss
plt.plot(epochs, validation_losses, 'ro-', label='Validation Loss')  # Red dots with line for validation loss
plt.xlabel('Epochs', fontsize=label_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)

# Display the plot
plt.show()
