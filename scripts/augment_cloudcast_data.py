import os
import numpy as np
from sklearn.preprocessing import StandardScaler # type: ignore
from skimage.transform import resize # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Paths for raw and processed data
RAW_TRAIN_DIR = './data/raw_data/cloudcast/CloudCastSmall/TrainCloud'
RAW_TEST_DIR = './data/raw_data/cloudcast/CloudCastSmall/TestCloud'
PROCESSED_TRAIN_DIR = './data/processed_data/cloudcast/train_unflattened'
PROCESSED_TEST_DIR = './data/processed_data/cloudcast/test_unflattened'
AUGMENTED_TRAIN_DIR = './data/processed_data/augmented_cloudcast/train'
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)
os.makedirs(AUGMENTED_TRAIN_DIR, exist_ok=True)

# ImageDataGenerator for augmentation (only for training data)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to load, resize, and normalize raw images (for both train and test data)
def preprocess_cloud_data(directory, target_size=(128, 128), output_dir=None):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    total_files = len(files)
    
    for i, file in enumerate(files):
        # Load raw image data
        file_data = np.load(os.path.join(directory, file))
        
        # Resize image
        resized_data = resize(file_data, target_size, anti_aliasing=True)
        
        # Normalize the image
        normalized_data = resized_data / 255.0
        
        # Save the preprocessed image
        output_file = os.path.join(output_dir, f"processed_image_{i}.npy")
        np.save(output_file, normalized_data)
        print(f"Processed image {i + 1}/{total_files} saved to {output_dir}")

# Function to apply augmentation to preprocessed training images
def augment_preprocessed_data(directory, batch_size=100, target_size=(128, 128), output_dir=None):
    files = [f for f in sorted(os.listdir(directory)) if f.endswith('.npy')]
    total_files = len(files)
    
    for i in range(0, total_files, batch_size):
        batch_files = files[i:i + batch_size]
        data_batch = []
        
        # Load preprocessed training data
        for file in batch_files:
            data_batch.append(np.load(os.path.join(directory, file)))
        
        data_batch = np.array(data_batch)
        
        # Reshape data into 128x128x3 for RGB (you may need to adjust if your data is grayscale)
        data_batch = data_batch.reshape((-1, *target_size, 1))  # For grayscale images
        
        # Apply augmentation
        augmented_data_batch = []
        for batch in datagen.flow(data_batch, batch_size=len(data_batch), shuffle=False):
            augmented_data_batch.append(batch)
            break  # Generate one batch of augmented data
        
        augmented_data_batch = np.array(augmented_data_batch).squeeze()
        
        # Save the augmented batch
        output_file = os.path.join(output_dir, f"augmented_batch_{i // batch_size}.npy")
        np.save(output_file, augmented_data_batch)
        print(f"Augmented batch {i // batch_size + 1} out of {total_files // batch_size + 1} saved.")

# Step 1: Preprocess raw train and test data
print("Preprocessing raw train data...")
preprocess_cloud_data(RAW_TRAIN_DIR, output_dir=PROCESSED_TRAIN_DIR)

print("Preprocessing raw test data (no augmentation)...")
preprocess_cloud_data(RAW_TEST_DIR, output_dir=PROCESSED_TEST_DIR)

# Step 2: Augment preprocessed train data only
print("Applying augmentation to preprocessed train data...")
augment_preprocessed_data(PROCESSED_TRAIN_DIR, output_dir=AUGMENTED_TRAIN_DIR)

print("Data preprocessing and augmentation complete!")
