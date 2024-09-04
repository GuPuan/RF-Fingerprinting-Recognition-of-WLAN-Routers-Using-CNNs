import matplotlib.pyplot as plt

# Read the data file
file_path = 'b2/b3_1.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists to hold the extracted data
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
        # Attempt to extract the second element and convert it to a float
        value = float(line.split()[1])
    except (IndexError, ValueError):
        continue  # Skip the line if parsing fails

    # Append the extracted value to the appropriate list
    if section == 'train_acc':
        train_accuracies.append(value)
    elif section == 'val_acc':
        validation_accuracies.append(value)
    elif section == 'train_loss':
        train_losses.append(value)
    elif section == 'val_loss':
        validation_losses.append(value)

# Ensure that all lists have the same length
min_length = min(len(train_accuracies), len(validation_accuracies), len(train_losses), len(validation_losses))
train_accuracies = train_accuracies[:min_length]
validation_accuracies = validation_accuracies[:min_length]
train_losses = train_losses[:min_length]
validation_losses = validation_losses[:min_length]

# Create a range of epochs for plotting
epochs = range(1, min_length + 1)

# Set up the figure and adjust the layout
plt.figure(figsize=(6, 6))  # Set the figure size to a more narrow format

# Define font sizes for labels, legend, and ticks
label_fontsize = 24
legend_fontsize = 22
ticks_fontsize = 24

# Adjust the spacing around the subplots
plt.subplots_adjust(left=0.2, right=0.4, top=0.8, bottom=0.45)  # Narrow the left and right margins

# Plot the accuracy graph (commented out, can be uncommented if needed)
# plt.plot(epochs, train_accuracies, 'bo-', label='Train Accuracy')  # Blue line with circle markers
# plt.plot(epochs, validation_accuracies, 'ro-', label='Validation Accuracy')  # Red line with circle markers
# plt.xlabel('Epochs', fontsize=label_fontsize)
# plt.ylabel('Accuracy', fontsize=label_fontsize)
# plt.legend(fontsize=legend_fontsize)
# plt.xticks(fontsize=ticks_fontsize)
# plt.yticks(fontsize=ticks_fontsize)

# Plot the loss graph
plt.plot(epochs, train_losses, 'bo-', label='Train Loss')  # Blue line with circle markers
plt.plot(epochs, validation_losses, 'ro-', label='Validation Loss')  # Red line with circle markers
plt.xlabel('Epochs', fontsize=label_fontsize)
plt.ylabel('Loss', fontsize=label_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)

# Display the plot
plt.show()
