# import os
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from skimage.transform import resize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # File paths for the CloudCast dataset
# RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
# RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'

# # Paths to GEO and TIMESTAMP files
# GEO_TRAIN_FILE = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud/GEO.npz'
# TIMESTAMPS_TRAIN_FILE = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud/TIMESTAMPS.npy'
# GEO_TEST_FILE = './data/raw_data/cloudcast/CloudCastSmall/TestCloud/GEO.npz'
# TIMESTAMPS_TEST_FILE = './data/raw_data/cloudcast/CloudCastSmall/TestCloud/TIMESTAMPS.npy'

# # Paths to save the processed data
# PROCESSED_TRAIN_DIR = './data/preprocessed_data/cloudcast/trainNew'
# PROCESSED_TEST_DIR = './data/preprocessed_data/cloudcast/testNew'

# os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
# os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# # Step 1: Load .npy files
# def load_data(directory):
#     files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
#     return files

# # Step 2: Handle missing data (already provided previously)
# def handle_missing_data():
#     def handle_missing_in_npy_files(input_dir, output_dir):
#         files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.npy')]
#         for file in files:
#             data = np.load(os.path.join(input_dir, file))
#             mask = np.isnan(data)
#             if mask.any():
#                 data[mask] = np.nanmean(data)
#             np.save(os.path.join(output_dir, file), data)

#     def handle_missing_in_npz(input_file, output_file):
#         data = np.load(input_file)
#         interpolated_data = {}
#         for key, array in data.items():
#             mask = np.isnan(array)
#             if mask.any():
#                 array[mask] = np.nanmean(array)
#             interpolated_data[key] = array
#         np.savez(output_file, **interpolated_data)

#     def handle_missing_in_timestamps(input_file, output_file):
#         data = np.load(input_file)
#         nans = np.isnan(data)
#         if nans.any():
#             data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), data[~nans])
#         np.save(output_file, data)

# # Step 3: Resize and normalize data
# def resize_and_normalize_data(directory, target_size=(128, 128)):
#     files = load_data(directory)
#     processed_data = []
#     for file in files:
#         file_data = np.load(os.path.join(directory, file))
#         resized_data = resize(file_data, target_size, anti_aliasing=True)
#         normalized_data = resized_data / 255.0
#         processed_data.append(normalized_data)
#     return np.array(processed_data)

# # Step 4: Convert to grayscale if using PCA
# def convert_to_grayscale(data):
#     grayscale_data = []
#     for img in data:
#         if img.ndim == 3:  # Check if image is RGB
#             grayscale_img = img.mean(axis=-1, keepdims=True)  # Convert to grayscale
#             grayscale_data.append(grayscale_img)
#         else:
#             grayscale_data.append(img)  # Already grayscale
#     return np.array(grayscale_data)

# # Step 5: Apply PCA
# def apply_pca(data, n_components=50):
#     scaler = StandardScaler()
#     data_flattened = data.reshape(data.shape[0], -1)  # Flatten the data
#     scaled_data = scaler.fit_transform(data_flattened)
#     max_components = min(n_components, scaled_data.shape[1])
#     pca = PCA(n_components=max_components)
#     pca_data = pca.fit_transform(scaled_data)
#     return pca_data

# # Step 6: Augment data using ImageDataGenerator
# def augment_data(data, target_size=(128, 128), output_dir=None):
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
#     augmented_data = []
#     for batch in datagen.flow(data, batch_size=len(data), shuffle=False):
#         augmented_data.append(batch)
#         break
#     return np.array(augmented_data).squeeze()

# # Step 7: Save data
# def save_data(data, output_dir, prefix='processed'):
#     for i, data_sample in enumerate(data):
#         output_file = os.path.join(output_dir, f"{prefix}_batch_{i}.npy")
#         np.save(output_file, data_sample)
#         print(f"Saved: {output_file}")

# # General preprocessing pipeline
# def preprocessing_pipeline(use_pca=False, use_augmentation=False, batch_size=100, target_size=(128, 128), grayscale_for_pca=False):
#     # Step 1: Load and preprocess train data
#     print("Loading and preprocessing training data...")
#     train_data = resize_and_normalize_data(RAW_TRAIN_DIR, target_size=target_size)
    
#     # Step 2: Convert to grayscale if PCA is enabled
#     if use_pca and grayscale_for_pca:
#         print("Converting training data to grayscale for PCA...")
#         train_data = convert_to_grayscale(train_data)
    
#     # Step 3: Apply PCA if enabled
#     if use_pca:
#         print(f"Applying PCA with {batch_size} components...")
#         train_data = apply_pca(train_data, n_components=batch_size)
    
#     # Step 4: Augment data if enabled
#     if use_augmentation:
#         print("Applying data augmentation...")
#         train_data = augment_data(train_data, target_size=target_size)
    
#     # Step 5: Save processed train data
#     save_data(train_data, PROCESSED_TRAIN_DIR, prefix='train')
    
#     # Repeat for test data (no augmentation for test)
#     print("Loading and preprocessing testing data...")
#     test_data = resize_and_normalize_data(RAW_TEST_DIR, target_size=target_size)
    
#     # Convert to grayscale if PCA is enabled for test data
#     if use_pca and grayscale_for_pca:
#         print("Converting testing data to grayscale for PCA...")
#         test_data = convert_to_grayscale(test_data)
    
#     # Apply PCA to test data if enabled
#     if use_pca:
#         print(f"Applying PCA to test data with {batch_size} components...")
#         test_data = apply_pca(test_data, n_components=batch_size)
    
#     # Step 5: Save processed test data
#     save_data(test_data, PROCESSED_TEST_DIR, prefix='test')
#     print("Data preprocessing complete!")

# # Run the pipeline with grayscale for PCA
# preprocessing_pipeline(use_pca=True, use_augmentation=True, batch_size=50, grayscale_for_pca=True)

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths for the CloudCast dataset
RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'
PROCESSED_TRAIN_DIR = './data/preprocessed_data/cloudcast/train'
PROCESSED_TEST_DIR = './data/preprocessed_data/cloudcast/test'

os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# Step 1: Load .npy files
def load_data(directory):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    data = [np.load(os.path.join(directory, file)) for file in files]
    return np.array(data)

# Step 2: Handle missing data
def handle_missing_data(data):
    mask = np.isnan(data)
    if mask.any():
        data[mask] = np.nanmean(data)
    return data

# Step 3: Resize and normalize data
def resize_and_normalize_data(data, target_size=(128, 128)):
    resized_data = np.array([resize(img, target_size, anti_aliasing=True) for img in data])
    normalized_data = resized_data / 255.0
    return normalized_data

# Step 4: Apply grayscale conversion for PCA
def convert_to_grayscale(data):
    grayscale_data = np.array([img.mean(axis=-1, keepdims=True) if img.ndim == 3 else img for img in data])
    return grayscale_data

# Step 5: Flatten and scale data
def flatten_and_scale_data(data):
    scaler = StandardScaler()
    data_flattened = data.reshape(data.shape[0], -1)
    scaled_data = scaler.fit_transform(data_flattened)
    return scaled_data

# Step 6: Apply PCA
def apply_pca(data, n_components=50):
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca_data

# Step 7: Augment data using ImageDataGenerator
def augment_data(data, target_size=(128, 128)):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    data = data.reshape((-1, *target_size, 1))  # Assuming grayscale images
    augmented_data = next(datagen.flow(data, batch_size=len(data), shuffle=False))
    return augmented_data

# Step 8: Save data
def save_data(data, output_dir, prefix='processed'):
    for i, data_sample in enumerate(data):
        output_file = os.path.join(output_dir, f"{prefix}_batch_{i}.npy")
        np.save(output_file, data_sample)
        print(f"Saved: {output_file}")

# General preprocessing pipeline
def preprocessing_pipeline(pca_option=False, augmentation_option=False, target_size=(128, 128), n_components=50):
    # Step 1: Load data
    print("Loading training and testing data...")
    train_data = load_data(RAW_TRAIN_DIR)
    test_data = load_data(RAW_TEST_DIR)

    # Step 2: Handle missing data
    print("Handling missing data...")
    train_data = handle_missing_data(train_data)
    test_data = handle_missing_data(test_data)

    # Step 3: Resize and normalize data
    print("Resizing and normalizing data...")
    train_data = resize_and_normalize_data(train_data, target_size)
    test_data = resize_and_normalize_data(test_data, target_size)

    # Option 1: PCA-based preprocessing
    if pca_option:
        print("Converting to grayscale for PCA...")
        train_data = convert_to_grayscale(train_data)
        test_data = convert_to_grayscale(test_data)

        print(f"Flattening, scaling, and applying PCA with {n_components} components...")
        train_data = flatten_and_scale_data(train_data)
        test_data = flatten_and_scale_data(test_data)

        train_data = apply_pca(train_data, n_components)
        test_data = apply_pca(test_data, n_components)

        # Save the PCA-processed data
        save_data(train_data, PROCESSED_TRAIN_DIR, prefix='train_pca')
        save_data(test_data, PROCESSED_TEST_DIR, prefix='test_pca')

    # Option 2: Augmentation-based preprocessing
    elif augmentation_option:
        print("Applying data augmentation...")
        train_data = augment_data(train_data, target_size)

        # Save augmented train data
        save_data(train_data, PROCESSED_TRAIN_DIR, prefix='train_augmented')

        # Test data (no augmentation, just resizing and normalizing)
        save_data(test_data, PROCESSED_TEST_DIR, prefix='test_normalized')

    print("Preprocessing complete!")

# Run the pipeline
# PCA-based preprocessing
print("Running PCA-based preprocessing...")
preprocessing_pipeline(pca_option=True, augmentation_option=False, target_size=(128, 128), n_components=50)

# Augmentation-based preprocessing
print("Running augmentation-based preprocessing...")
preprocessing_pipeline(pca_option=False, augmentation_option=True, target_size=(128, 128))
