import numpy as np
import matplotlib.pyplot as plt

file_path = 'extracted_data/6_0.19_0.8_-39/segment_600.npy'

# Function to read OFDM data from a .npy file
def read_OFDM_data(file_path):
    return np.load(file_path)  # Load the numpy array from the specified file

# Function to print the OFDM data
def print_OFDM_data(OFDM_data):
    print("OFDM Data (Real part, Imaginary part):")
    print(OFDM_data)  # Print the OFDM data in the form of real and imaginary parts

# Function to visualize the OFDM data
def visualize_OFDM_data(OFDM_data):
    plt.figure(figsize=(12, 8))  # Set the figure size

    # Plot the real part of the OFDM data
    plt.subplot(2, 1, 1)  # Create a subplot for the real part
    plt.plot(OFDM_data[:, 0], 'o-', label='Real Part')  # Plot real part with markers
    plt.legend()  # Show the legend
    plt.xlabel('Index')  # Set the x-axis label
    plt.ylabel('Amplitude')  # Set the y-axis label
    plt.title('Real Part of OFDM Data')  # Set the title

    # Plot the imaginary part of the OFDM symbol
    plt.subplot(2, 1, 2)  # Create a subplot for the imaginary part
    plt.plot(OFDM_data[:, 1], 'o-', label='Imaginary Part')  # Plot imaginary part with markers
    plt.legend()  # Show the legend
    plt.xlabel('Index')  # Set the x-axis label
    plt.ylabel('Amplitude')  # Set the y-axis label
    plt.title('Imaginary Part of OFDM Data')  # Set the title

    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.show()  # Display the plots

# Read the OFDM data from the file
OFDM_data = read_OFDM_data(file_path)

# Print the OFDM data
print_OFDM_data(OFDM_data)

# Visualize the OFDM data
visualize_OFDM_data(OFDM_data)
