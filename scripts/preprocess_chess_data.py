import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Paths
RAW_DATA_PATH = "./data/raw_data/chess/games.csv"
PROCESSED_TRAIN_PATH = "./data/processed_data/chess/train"
PROCESSED_TEST_PATH = "./data/processed_data/chess/test"

# Ensure directories exist
os.makedirs(PROCESSED_TRAIN_PATH, exist_ok=True)
os.makedirs(PROCESSED_TEST_PATH, exist_ok=True)

# Load data
df = pd.read_csv(RAW_DATA_PATH)

# Feature Engineering: Define important features
features = [
    'rated', 'turns', 'victory_status', 'increment_code', 'white_rating', 
    'black_rating', 'opening_eco', 'opening_ply'
]
target = 'winner'  # Target variable

# Encode categorical features using One-Hot Encoding
categorical_features = ['victory_status', 'increment_code', 'opening_eco']
df_encoded = pd.get_dummies(df[features], columns=categorical_features)

# Encode target (winner)
target_map = {'white': 0, 'black': 1, 'draw': 2}
df[target] = df[target].map(target_map)

# Split dataset into training, validation, and test sets
X = df_encoded.values
y = df[target].values

# Perform Train-Test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Further split the training data into train and validation sets (80% train, 20% validation from train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize the numerical features (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Save preprocessed data as .npy files
np.save(os.path.join(PROCESSED_TRAIN_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(PROCESSED_TRAIN_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(PROCESSED_TRAIN_PATH, 'X_val.npy'), X_val)
np.save(os.path.join(PROCESSED_TRAIN_PATH, 'y_val.npy'), y_val)
np.save(os.path.join(PROCESSED_TEST_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(PROCESSED_TEST_PATH, 'y_test.npy'), y_test)

print("Preprocessing and splitting completed. Data saved as .npy files.")
