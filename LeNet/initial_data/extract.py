import os
import shutil
import numpy as np

input_dir = 'ori_data'  # Directory containing the original data files
output_base_dir = 'extracted_data'  # Directory to save the extracted segments

def save_complex_as_2d(input_file, output_dir, num_segments=1000):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the input file as an array of complex numbers
    data = np.fromfile(input_file, dtype=np.complex64)
    segments = []  # List to hold the valid segments
    segment = []  # Temporary list to build a segment

    # Iterate through the data to extract segments of non-zero values
    for value in data:
        if value != 0:
            segment.append(value)  # Add non-zero values to the current segment
        else:
            if len(segment) >= 128:  # Check if the segment has at least 128 values
                segments.append(np.array(segment[:128]))  # Save the first 128 values of the segment
                if len(segments) == num_segments:  # Stop if we have enough segments
                    break
            segment = []  # Reset the segment list for the next segment

    # Check and save the last segment if it meets the criteria
    if len(segment) >= 128 and len(segments) < num_segments:
        segments.append(np.array(segment[:128]))

    # Save each segment as a 2D numpy array (real and imaginary parts)
    for i, segment in enumerate(segments):
        iq_array = np.column_stack((segment.real, segment.imag))  # Create a 2D array with real and imaginary parts
        output_file = os.path.join(output_dir, f'segment_{i + 1}.npy')  # Name the output file
        np.save(output_file, iq_array)  # Save the array as a .npy file
        print(f"Saved segment {i + 1} data to {output_file}")

    print(f"Total {len(segments)} segments saved to {output_dir}")

def process_all_files(input_dir, output_base_dir):
    # Check if the extracted_data directory exists; delete it if it does, or create it if it doesn't
    if os.path.exists(output_base_dir):
        shutil.rmtree(output_base_dir)
    os.makedirs(output_base_dir)

    # Iterate through the files in the ori_data directory
    for i, file_name in enumerate(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            # Rename the folder for each file to "i_" + original file name, with i starting from 0
            new_folder_name = f"{i}_{os.path.splitext(file_name)[0]}"
            output_dir = os.path.join(output_base_dir, new_folder_name)
            save_complex_as_2d(file_path, output_dir)  # Process and save the file

process_all_files(input_dir, output_base_dir)
