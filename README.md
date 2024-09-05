# README

Welcome to the Radio Frequency Fingerprinting Recognition of
WLAN Routers Using Convolutional Networks Project! This supporting material is organized into two main parts: one for generating WiFi data frames using GNURadio, and another for training neural networks using the generated data.

## Project Structure

### 1. **GRC Files for GNURadio (grcLTF Folder)**

This folder contains two essential GRC files for generating WiFi data frames:

- **wifi_loopback.grc**
- **wifi_phy_hier.grc**

#### How to Use:

1. **Initial Setup:**
   If this is your first time running the project, you need to set the file path for data collection in the `file sink` block located after the `WiFi Sync Long` block in the `wifi_phy_hier.grc` file. Set the path to the `ori_data` folder as follows:
   `'/home/buan/Supporting Material/LeNet/initial_data/ori_data/'+phase_noise_string+'_'+fre_offset_string+'_'+DC_offset_string`
2. **Run the Scripts:**
   Once the path is correctly set, open and run the `wifi_loopback.grc` file in GNURadio Companion to start collecting WiFi data frames.

### 2. **Neural Network Training Files**

This section contains scripts for training neural networks, including LeNet and GoogLeNet. The process for both models is similar; however, the instructions below focus on LeNet as an example.

#### Steps to Train the Neural Network:

1. **Data Extraction:**

- Run `extract.py` to extract LTF OFDM symbols from the collected data frames stored in the `ori_data` folder under `initial_data`.
- The extracted data will be saved in the `extracted_data` folder.

2. **Data Partitioning:**

- Run `data_partitioning.py` to split the dataset into training and testing sets.
- The partitioned data will be saved in the `mode_data` folder.

3. **Model Training:**

- Run `model_train.py` to start training the model. You can set the number of epochs and the path for the log file within this script.
- Example code:
  ```python
  train_process = train_model_process(LeNet, train_data, val_data, num_epochs=50)
  log_file = open("b4_20.txt", "w")
  ```

4. **Model Testing:**

- After training, run `model_test.py` to validate the model using the testing set and obtain accuracy metrics.

### Additional Files:

1. **LTF Signal Plotting:**

- The `LTF_plot.py`  under `initial_data` contains scripts to plot the IQ signals and OFDM symbols of the LTF.

2. **Result Visualization:**

- The `Result` folder stores data results, with each subfolder corresponding to results from different convolutional or pooling layer changes.
- The `view_complexNpy.py` script can be used to view the extracted LTF OFDM symbols saved in `.npy` format.
- Use `viewResult.py` to visualize the training results for a specific round.
- The `LeNetAverageResult.py` in the `Result` folder contains visualizations of the aggregated results.

## Contact Information

If you have any questions or need further assistance, feel free to reach out to me at:
**Email:** [b.gu@hss23.qmul.ac.uk](mailto:b.gu@hss23.qmul.ac.uk)

Thank you for using this project!

