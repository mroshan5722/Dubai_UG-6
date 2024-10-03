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
#--------------------------------------------------------------------------------------------------------------------------------------------------------------#
# import os
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from skimage.transform import resize
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Paths for the raw and processed CloudCast dataset
# RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
# RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'
# PROCESSED_TRAIN_DIR = './data/preprocessed_data/cloudcast/trainNew'
# PROCESSED_TEST_DIR = './data/preprocessed_data/cloudcast/testNew'
# os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
# os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)


# def load_data(directory):
#     files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
#     data = []
    
#     # Loop through each file, load the data and check its shape
#     for file in files:
#         file_data = np.load(os.path.join(directory, file))
        
#         if len(file_data.shape) == 2:  # Check if the file has 2 dimensions (consistent)
#             data.append(file_data)
#         else:
#             print(f"Skipping {file}: Inconsistent shape {file_data.shape}")
    
#     return np.array(data, dtype=object)  # Use 'dtype=object' to handle any mixed shapes



# # Function to handle missing data in npy files
# def handle_missing_in_npy_files(input_dir, output_dir):
#     files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.npy')]
#     for file in files:
#         data = np.load(os.path.join(input_dir, file))
#         data = data.astype(np.float32)  # Ensuring the data is in a float32 format for compatibility
#         mask = np.isnan(data)
#         if mask.any():
#             data[mask] = np.nanmean(data)
#         np.save(os.path.join(output_dir, file), data)

# # Function to handle missing data in npz files
# def handle_missing_in_npz(input_file, output_file):
#     data = np.load(input_file)
#     interpolated_data = {}
#     for key, array in data.items():
#         array = array.astype(np.float32)
#         mask = np.isnan(array)
#         if mask.any():
#             array[mask] = np.nanmean(array)
#         interpolated_data[key] = array
#     np.savez(output_file, **interpolated_data)

# # Function to handle missing data in timestamp files
# def handle_missing_in_timestamps(input_file, output_file):
#     data = np.load(input_file)
#     data = data.astype(np.float32)
#     nans = np.isnan(data)
#     if nans.any():
#         data[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), data[~nans])
#     np.save(output_file, data)


# # Step 3: Resize and normalize data
# def resize_and_normalize_data(data, target_size=(128, 128)):
#     resized_data = np.array([resize(img, target_size, anti_aliasing=True) for img in data])
#     normalized_data = resized_data / 255.0
#     return normalized_data

# # Step 4: Apply grayscale conversion for PCA
# def convert_to_grayscale(data):
#     grayscale_data = np.array([img.mean(axis=-1, keepdims=True) if img.ndim == 3 else img for img in data])
#     return grayscale_data

# # Step 5: Flatten and scale data
# def flatten_and_scale_data(data):
#     scaler = StandardScaler()
#     data_flattened = data.reshape(data.shape[0], -1)
#     scaled_data = scaler.fit_transform(data_flattened)
#     return scaled_data

# # Step 6: Apply PCA
# def apply_pca(data, n_components=50):
#     pca = PCA(n_components=n_components)
#     pca_data = pca.fit_transform(data)
#     return pca_data

# # Step 7: Augment data using ImageDataGenerator
# def augment_data(data, target_size=(128, 128)):
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
#     data = data.reshape((-1, *target_size, 1))  # Assuming grayscale images
#     augmented_data = next(datagen.flow(data, batch_size=len(data), shuffle=False))
#     return augmented_data

# # Step 8: Save data
# def save_data(data, output_dir, prefix='processed'):
#     for i, data_sample in enumerate(data):
#         output_file = os.path.join(output_dir, f"{prefix}_batch_{i}.npy")
#         np.save(output_file, data_sample)
#         print(f"Saved: {output_file}")

# # General preprocessing pipeline
# def preprocessing_pipeline(pca_option=False, augmentation_option=False, target_size=(128, 128), n_components=50):
#     # Step 1: Load data
#     print("Loading training and testing data...")
#     train_data = load_data(RAW_TRAIN_DIR)
#     test_data = load_data(RAW_TEST_DIR)

#     # Step 2: Handle missing data
#     # Handle missing data in train and test npy files
#     handle_missing_in_npy_files(RAW_TRAIN_DIR, PROCESSED_TRAIN_DIR)
#     handle_missing_in_npy_files(RAW_TEST_DIR, PROCESSED_TEST_DIR)
    
#     # Handle missing data in GEO and TIMESTAMP files (train and test)
#     handle_missing_in_npz('./data/raw_data/cloudcast/CloudCastSmall/TrainCloud/GEO.npz', './data/preprocessed_data/cloudcast/trainNew/GEO.npz')
#     handle_missing_in_npz('./data/raw_data/cloudcast/CloudCastSmall/TestCloud/GEO.npz', './data/preprocessed_data/cloudcast/testNew/GEO.npz')
#     handle_missing_in_timestamps('./data/raw_data/cloudcast/CloudCastSmall/TrainCloud/TIMESTAMPS.npy', './data/preprocessed_data/cloudcast/trainNew/TIMESTAMPS.npy')
#     handle_missing_in_timestamps('./data/raw_data/cloudcast/CloudCastSmall/TestCloud/TIMESTAMPS.npy', './data/preprocessed_data/cloudcast/testNew/TIMESTAMPS.npy')

#     # Step 3: Resize and normalize data
#     print("Resizing and normalizing data...")
#     train_data = resize_and_normalize_data(train_data, target_size)
#     test_data = resize_and_normalize_data(test_data, target_size)

#     # Option 1: PCA-based preprocessing
#     if pca_option:
#         print("Converting to grayscale for PCA...")
#         train_data = convert_to_grayscale(train_data)
#         test_data = convert_to_grayscale(test_data)

#         print(f"Flattening, scaling, and applying PCA with {n_components} components...")
#         train_data = flatten_and_scale_data(train_data)
#         test_data = flatten_and_scale_data(test_data)

#         train_data = apply_pca(train_data, n_components)
#         test_data = apply_pca(test_data, n_components)

#         # Save the PCA-processed data
#         save_data(train_data, PROCESSED_TRAIN_DIR, prefix='train_pca')
#         save_data(test_data, PROCESSED_TEST_DIR, prefix='test_pca')

#     # Option 2: Augmentation-based preprocessing
#     elif augmentation_option:
#         print("Applying data augmentation...")
#         train_data = augment_data(train_data, target_size)

#         # Save augmented train data
#         save_data(train_data, PROCESSED_TRAIN_DIR, prefix='train_augmented')

#         # Test data (no augmentation, just resizing and normalizing)
#         save_data(test_data, PROCESSED_TEST_DIR, prefix='test_normalized')

#     print("Preprocessing complete!")

# # Example of running the pipeline
# # Choose between PCA or Augmentation by setting respective flags to True
# preprocessing_pipeline(pca_option=True, augmentation_option=False, n_components=50)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------#
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths for the raw and processed CloudCast dataset
RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'
PROCESSED_TRAIN_DIR = './data/preprocessed_data/cloudcast/trainNew'
PROCESSED_TEST_DIR = './data/preprocessed_data/cloudcast/testNew'
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# Function to load data in batches
def load_data_in_batches(directory, batch_size):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        # Add allow_pickle=True when loading .npy files
        data_batch = [np.load(os.path.join(directory, file), allow_pickle=True) for file in batch_files]
        yield np.array(data_batch, dtype=object), batch_files  # Returning file names as well


# Function to handle missing data in batches
def handle_missing_in_npy_files(input_dir, output_dir, batch_size):
    for batch_index, (data_batch, batch_files) in enumerate(load_data_in_batches(input_dir, batch_size)):
        print(f"Processing batch {batch_index + 1} for missing data handling...")
        for i, file_data in enumerate(data_batch):
            file_data = file_data.astype(np.float32)
            mask = np.isnan(file_data)
            if mask.any():
                file_data[mask] = np.nanmean(file_data)
            np.save(os.path.join(output_dir, batch_files[i]), file_data)
        print(f"Batch {batch_index + 1} missing data handled and saved.")

# Resize and normalize data in batches
def resize_and_normalize_data_in_batches(input_dir, output_dir, target_size, batch_size):
    for data_batch, batch_files in load_data_in_batches(input_dir, batch_size):
        processed_batch = []
        for img in data_batch:
            # Ensure the image is in float32 format
            img = img.astype(np.float32)
            resized_data = resize(img, target_size, anti_aliasing=True)
            normalized_data = resized_data / 255.0
            processed_batch.append(normalized_data)
        processed_batch = np.array(processed_batch)
        # Save the batch
        for i, img_data in enumerate(processed_batch):
            np.save(os.path.join(output_dir, batch_files[i]), img_data)
            print(f"Processed and saved: {batch_files[i]}")


# Apply grayscale conversion for PCA in batches
def convert_to_grayscale_in_batches(input_dir, output_dir, batch_size):
    for batch_index, (data_batch, batch_files) in enumerate(load_data_in_batches(input_dir, batch_size)):
        print(f"Processing batch {batch_index + 1} for grayscale conversion...")
        grayscale_batch = []
        for img in data_batch:
            if img.ndim == 3:  # Convert to grayscale
                grayscale_img = img.mean(axis=-1, keepdims=True)
            else:
                grayscale_img = img
            grayscale_batch.append(grayscale_img)
        # Save the grayscale batch
        for i, img_data in enumerate(grayscale_batch):
            np.save(os.path.join(output_dir, batch_files[i]), img_data)
        print(f"Batch {batch_index + 1} converted to grayscale and saved.")

# Flatten, scale, and apply PCA in batches
def flatten_scale_and_apply_pca_in_batches(input_dir, output_dir, n_components, batch_size):
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)
    
    for batch_index, (data_batch, batch_files) in enumerate(load_data_in_batches(input_dir, batch_size)):
        print(f"Processing batch {batch_index + 1} for flattening, scaling, and PCA...")
        data_batch_flattened = [img.reshape(-1) for img in data_batch]
        scaled_data = scaler.fit_transform(data_batch_flattened)
        pca_data = pca.fit_transform(scaled_data)
        
        # Save PCA-transformed batch
        for i, pca_img in enumerate(pca_data):
            np.save(os.path.join(output_dir, batch_files[i]), pca_img)
        print(f"Batch {batch_index + 1} PCA applied and saved.")

# Augment data using ImageDataGenerator in batches
def augment_data_in_batches(input_dir, output_dir, target_size, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    for batch_index, (data_batch, batch_files) in enumerate(load_data_in_batches(input_dir, batch_size)):
        print(f"Processing batch {batch_index + 1} for augmentation...")
        data_batch = data_batch.reshape((-1, *target_size, 1))  # Assuming grayscale images
        augmented_batch = next(datagen.flow(data_batch, batch_size=len(data_batch), shuffle=False))
        
        # Save augmented batch
        for i, aug_img in enumerate(augmented_batch):
            np.save(os.path.join(output_dir, batch_files[i]), aug_img)
        print(f"Batch {batch_index + 1} augmented and saved.")

# General preprocessing pipeline
def preprocessing_pipeline(pca_option=False, augmentation_option=False, target_size=(128, 128), n_components=50, batch_size=100):
    # Step 1: Handle missing data
    print("Handling missing data for training and testing...")
    handle_missing_in_npy_files(RAW_TRAIN_DIR, PROCESSED_TRAIN_DIR, batch_size)
    handle_missing_in_npy_files(RAW_TEST_DIR, PROCESSED_TEST_DIR, batch_size)

    # Step 2: Resize and normalize data
    print("Resizing and normalizing data...")
    resize_and_normalize_data_in_batches(PROCESSED_TRAIN_DIR, PROCESSED_TRAIN_DIR, target_size, batch_size)
    resize_and_normalize_data_in_batches(PROCESSED_TEST_DIR, PROCESSED_TEST_DIR, target_size, batch_size)

    # Option 1: PCA-based preprocessing
    if pca_option:
        print("Converting to grayscale and applying PCA...")
        convert_to_grayscale_in_batches(PROCESSED_TRAIN_DIR, PROCESSED_TRAIN_DIR, batch_size)
        convert_to_grayscale_in_batches(PROCESSED_TEST_DIR, PROCESSED_TEST_DIR, batch_size)
        
        flatten_scale_and_apply_pca_in_batches(PROCESSED_TRAIN_DIR, PROCESSED_TRAIN_DIR, n_components, batch_size)
        flatten_scale_and_apply_pca_in_batches(PROCESSED_TEST_DIR, PROCESSED_TEST_DIR, n_components, batch_size)

    # Option 2: Augmentation-based preprocessing
    elif augmentation_option:
        print("Applying data augmentation...")
        augment_data_in_batches(PROCESSED_TRAIN_DIR, PROCESSED_TRAIN_DIR, target_size, batch_size)

    print("Batch preprocessing complete!")

# Example of running the pipeline
preprocessing_pipeline(pca_option=True, augmentation_option=False, n_components=50, batch_size=100)
