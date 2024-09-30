import os
import numpy as np

# Paths to the raw CloudCast dataset
RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'

# Paths to GEO and TIMESTAMP files
GEO_TRAIN_FILE = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud/GEO.npz'
TIMESTAMPS_TRAIN_FILE = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud/TIMESTAMPS.npy'
GEO_TEST_FILE = './data/raw_data/cloudcast/CloudCastSmall/TestCloud/GEO.npz'
TIMESTAMPS_TEST_FILE = './data/raw_data/cloudcast/CloudCastSmall/TestCloud/TIMESTAMPS.npy'

def check_missing_in_npy_files(directory):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    
    for file in files:
        data = np.load(os.path.join(directory, file))
        if np.isnan(data).any():
            return "Yes"
    return "No"

def check_missing_in_npz(file_path):
    data = np.load(file_path)
    for key, array in data.items():
        if np.isnan(array).any():
            return "Yes"
    return "No"

def check_missing_in_metadata(file_path):
    data = np.load(file_path)
    if np.isnan(data).any():
        return "Yes"
    return "No"

# Check missing values in raw train and test datasets
print("Checking missing values in TrainCloud images:", check_missing_in_npy_files(RAW_TRAIN_DIR))
print("Checking missing values in TestCloud images:", check_missing_in_npy_files(RAW_TEST_DIR))

# Check missing values in GEO and TIMESTAMP metadata
print("Checking missing values in GEO Train metadata:", check_missing_in_npz(GEO_TRAIN_FILE))
print("Checking missing values in GEO Test metadata:", check_missing_in_npz(GEO_TEST_FILE))
print("Checking missing values in TIMESTAMPS Train:", check_missing_in_metadata(TIMESTAMPS_TRAIN_FILE))
print("Checking missing values in TIMESTAMPS Test:", check_missing_in_metadata(TIMESTAMPS_TEST_FILE))
