import os
import numpy as np
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore

# Define paths for the CloudCastSmall dataset
RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'
PROCESSED_TRAIN_DIR = './data/processed_data/cloudcast/train'
PROCESSED_TEST_DIR = './data/processed_data/cloudcast/test'
GEO_TRAIN_FILE = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud/GEO.npz'
TIMESTAMPS_TRAIN_FILE = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud/TIMESTAMPS.npy'
GEO_TEST_FILE = './data/raw_data/cloudcast/CloudCastSmall/TestCloud/GEO.npz'
TIMESTAMPS_TEST_FILE = './data/raw_data/cloudcast/CloudCastSmall/TestCloud/TIMESTAMPS.npy'

# Ensure processed data directories exist
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# Function to load cloud data from .npy files
def load_cloud_data(directory, limit=None):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    if limit:
        files = files[:limit]
    
    data = [np.load(os.path.join(directory, file)) for file in files]
    return np.array(data)

# Preprocessing pipeline for CloudCast dataset
def preprocess_cloudcast_data(data, apply_pca=False):
    # 1. Normalize pixel values to [0, 1]
    data = data / 255.0
    
    # 2. Flatten images (required for models like KNN, MLP, etc.)
    flattened_data = data.reshape(data.shape[0], -1)
    
    # 3. Standardize features (especially for clustering and non-CNN models)
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(flattened_data)
    
    # 4. Apply PCA for dimensionality reduction (Optional, for clustering)
    if apply_pca:
        pca = PCA(n_components=50)  # Adjust the number of components as needed
        data_pca = pca.fit_transform(standardized_data)
        return data_pca
    
    return standardized_data

# Load train and test cloud data
print("Loading cloud data...")
train_data = load_cloud_data(RAW_TRAIN_DIR)
test_data = load_cloud_data(RAW_TEST_DIR)

# Preprocess the train and test data
print("Preprocessing train and test cloud data...")
processed_train_data = preprocess_cloudcast_data(train_data, apply_pca=True)
processed_test_data = preprocess_cloudcast_data(test_data, apply_pca=True)

# Save processed train and test data
np.save(os.path.join(PROCESSED_TRAIN_DIR, 'processed_train_cloudcast.npy'), processed_train_data)
np.save(os.path.join(PROCESSED_TEST_DIR, 'processed_test_cloudcast.npy'), processed_test_data)

print("Data preprocessing complete!")
