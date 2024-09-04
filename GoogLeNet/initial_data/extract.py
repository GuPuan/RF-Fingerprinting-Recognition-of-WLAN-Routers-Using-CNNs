import os
import shutil
import numpy as np

# Define input and output directories
input_dir = 'ori_data'
output_base_dir = 'extracted_data'  # Output folder within this project

def save_complex_as_2d(input_file, output_dir, num_segments=1000):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load complex data from the binary file
    data = np.fromfile(input_file, dtype=np.complex64)
    segments = []
    segment = []

    # Segment the data into arrays of length 128
    for value in data:
        if value != 0:
            segment.append(value)
        else:
            # Save segments that have at least 128 non-zero values
            if len(segment) >= 128:
                segments.append(np.array(segment[:128]))
                if len(segments) == num_segments:  # Stop if the desired number of segments is reached
                    break
            segment = []

    # Add any remaining segments if they meet the size criteria
    if len(segment) >= 128 and len(segments) < num_segments:
        segments.append(np.array(segment[:128]))

    # Save each segment as a .npy file with real and imaginary parts as separate columns
    for i, segment in enumerate(segments):
        iq_array = np.column_stack((segment.real, segment.imag))
        output_file = os.path.join(output_dir, f'segment_{i + 1}.npy')
        np.save(output_file, iq_array)
        print(f"Saved segment {i + 1} to {output_file}")

    print(f"Total of {len(segments)} segments saved to {output_dir}")

def process_all_files(input_dir, output_base_dir):
    # Check if the extracted_data folder exists; if it does, delete it. Then create a new one.
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)

    # Process each file in the ori_data directory, rename the folders, and save the data
    for i, file_name in enumerate(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            new_folder_name = f"{i}_{os.path.splitext(file_name)[0]}"
            output_dir = os.path.join(output_base_dir, new_folder_name)
            save_complex_as_2d(file_path, output_dir)

# Execute the processing function
process_all_files(input_dir, output_base_dir)
