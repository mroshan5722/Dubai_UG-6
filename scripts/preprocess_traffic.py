import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.io import imsave

# Paths for raw and processed dataset
RAW_TRAIN_DIR = '../data/raw_data/traffic/traffic_Data/DATA/'
RAW_TEST_DIR = '../data/raw_data/traffic/traffic_Data/TEST/'
PROCESSED_TRAIN_DIR = '../data/preprocessed_data/traffic/trainNew/'
PROCESSED_TEST_DIR = '../data/preprocessed_data/traffic/testNew/'

# Parameters
image_size = (100, 100)
num_classes = len(os.listdir(RAW_TRAIN_DIR))  # Total number of classes

# Create directories for processed data
print(f"Creating directories for processed data...")
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

# Preprocessing functions
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, image_size)

def equalize(img):
    return cv2.equalizeHist(img)

def normalize(img):
    return img / 255.0

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = normalize(img)
    return img

# Augmentation generator for train data
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)

# Process training data
print("Processing training data...")
train_images = []
train_labels = []

for label, class_name in enumerate(os.listdir(RAW_TRAIN_DIR)):
    class_dir = os.path.join(RAW_TRAIN_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"Skipping non-directory: {class_dir}")
        continue

    save_class_dir = os.path.join(PROCESSED_TRAIN_DIR, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
        processed_img = preprocessing(img)
        train_images.append(processed_img)
        train_labels.append(label)
        save_path = os.path.join(save_class_dir, f"preprocessed_{img_name}")
        if not cv2.imwrite(save_path, (processed_img * 255).astype(np.uint8)):
            print(f"Failed to save image: {save_path}")
        else:
            print(f"Saved image: {save_path}")

train_images = np.array(train_images).reshape(-1, *image_size, 1)
train_labels = np.array(train_labels)
print(f"Training data processed: {train_images.shape[0]} images")

# Augment training data
datagen.fit(train_images)

# Process test data
print("Processing test data...")
test_images = []

for img_name in os.listdir(RAW_TEST_DIR):
    img_path = os.path.join(RAW_TEST_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read test image: {img_path}")
        continue
    processed_img = preprocessing(img)
    test_images.append(processed_img)
    save_path = os.path.join(PROCESSED_TEST_DIR, f"preprocessed_{img_name}")
    if not cv2.imwrite(save_path, (processed_img * 255).astype(np.uint8)):
        print(f"Failed to save test image: {save_path}")
    else:
        print(f"Saved test image: {save_path}")

test_images = np.array(test_images).reshape(-1, *image_size, 1)
print(f"Test data processed: {test_images.shape[0]} images")

print("Data processing complete!")
