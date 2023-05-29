# Import libraries
import os
import numpy as np
import pandas as pd
from edf_reader import read_edf_file

# Define folders and file extension
folder1 = "datasets/EDF-D"  # Change this to the name of the first folder
folder2 = "datasets/EDF-ND"  # Change this to the name of the second folder
ext = ".edf"  # Change this to the file extension

# Define a function to regularize data to (-1,+1)


def regularize(data):
    # Find the minimum and maximum values of the data
    min_val = np.min(data)
    max_val = np.max(data)
    # Subtract the minimum and divide by the range
    data = (data - min_val) / (max_val - min_val)
    # Scale to (-1,+1) by multiplying by 2 and subtracting 1
    data = data * 2 - 1
    return data


# Initialize empty lists to store the data of each class and each signal
data1_1 = []  # Data of class 1 and signal 1
data1_2 = []  # Data of class 1 and signal 2
data1_3 = []  # Data of class 1 and signal 3
data1_4 = []  # Data of class 1 and signal 4
data2_1 = []  # Data of class 2 and signal 1
data2_2 = []  # Data of class 2 and signal 2
data2_3 = []  # Data of class 2 and signal 3
data2_4 = []  # Data of class 2 and signal 4

# Iterate through the files in the first folder
for file in os.listdir(folder1):
    # Check if the file has the right extension
    if file.endswith(ext):
        # Get the file path
        file_path = os.path.join(folder1, file)
        # Read the edf file using read_edf_file function
        sigbufs, signal_labels = read_edf_file(file_path)
        # Regularize each signal in sigbufs
        sigbufs = regularize(sigbufs)
        # Concate each signal to the corresponding list of the first class
        data1_1 = np.concatenate((data1_1, sigbufs[0, :]))
        data1_2 = np.concatenate((data1_2, sigbufs[1, :]))
        data1_3 = np.concatenate((data1_3, sigbufs[2, :]))
        data1_4 = np.concatenate((data1_4, sigbufs[3, :]))

# Iterate through the files in the second folder
for file in os.listdir(folder2):
    # Check if the file has the right extension
    if file.endswith(ext):
        # Get the file path
        file_path = os.path.join(folder2, file)
        # Read the edf file using read_edf_file function
        sigbufs, signal_labels = read_edf_file(file_path)
        # Regularize each signal in sigbufs
        sigbufs = regularize(sigbufs)
        # Concate each signal to the corresponding list of the second class
        data2_1 = np.concatenate((data2_1, sigbufs[0, :]))
        data2_2 = np.concatenate((data2_2, sigbufs[1, :]))
        data2_3 = np.concatenate((data2_3, sigbufs[2, :]))
        data2_4 = np.concatenate((data2_4, sigbufs[3, :]))

# Convert the lists of data to pandas dataframes with four columns each
data1 = pd.DataFrame({"EEG L1(Fp1)": data1_1, "EEG R1(Fp2)": data1_2,
                     "EEG L2(F7)": data1_3, "EEG R2(F8)": data1_4})
data2 = pd.DataFrame({"EEG L1(Fp1)": data2_1, "EEG R1(Fp2)": data2_2,
                     "EEG L2(F7)": data2_3, "EEG R2(F8)": data2_4})

# Save the dataframes as csv files
data1.to_csv("data-d.csv", index=False)
data2.to_csv("data-nd.csv", index=False)
