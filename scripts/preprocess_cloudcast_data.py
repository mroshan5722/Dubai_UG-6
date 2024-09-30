import os
import numpy as np
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.decomposition import PCA # type: ignore
from skimage.transform import resize # type: ignore

# Define paths for the CloudCastSmall dataset
RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'
PROCESSED_TRAIN_DIR = './data/processed_data/cloudcast/train'
PROCESSED_TEST_DIR = './data/processed_data/cloudcast/test'

# Ensure processed data directories exist
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# Function to load and preprocess cloud data in batches
def process_data_in_batches(directory, batch_size=100, target_size=(128, 128), output_dir=None, apply_pca=True):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    scaler = StandardScaler()
    total_files = len(files)

    for i in range(0, total_files, batch_size):
        batch_files = files[i:i + batch_size]
        data_batch = []
        
        # Load and resize data in batch
        for file in batch_files:
            file_data = np.load(os.path.join(directory, file))
            resized_data = resize(file_data, target_size, anti_aliasing=True)
            data_batch.append(resized_data)
        
        data_batch = np.array(data_batch)
        
        # Normalize and flatten the batch
        data_batch = data_batch / 255.0
        flattened_data_batch = data_batch.reshape(data_batch.shape[0], -1)
        
        # Standardize the batch
        standardized_data_batch = scaler.fit_transform(flattened_data_batch)
        
        # Apply PCA for dimensionality reduction if enabled
        if apply_pca:
            # Calculate the maximum allowable components for PCA
            max_components = min(standardized_data_batch.shape[0], standardized_data_batch.shape[1])
            pca = PCA(n_components=min(50, max_components))  # Adjust components dynamically
            standardized_data_batch = pca.fit_transform(standardized_data_batch)
        
        # Save each batch
        output_file = os.path.join(output_dir, f"processed_batch_{i // batch_size}.npy")
        np.save(output_file, standardized_data_batch)
        print(f"Processed batch {i // batch_size + 1} out of {total_files // batch_size + 1}")

# Process train and test data in batches
print("Processing train data in batches...")
process_data_in_batches(RAW_TRAIN_DIR, batch_size=100, output_dir=PROCESSED_TRAIN_DIR)

print("Processing test data in batches...")
process_data_in_batches(RAW_TEST_DIR, batch_size=100, output_dir=PROCESSED_TEST_DIR)

print("Data processing complete!")
