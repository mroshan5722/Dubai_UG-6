import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from sklearn.decomposition import PCA # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from scipy import stats # type: ignore

# File paths for the raw data and processed data directories
RAW_DATA_PATH = './data/raw_data/phishing/out.csv'
PROCESSED_TRAIN_FLATTENED_DIR = './data/processed_data/phishing/train_flattened'
PROCESSED_TEST_FLATTENED_DIR = './data/processed_data/phishing/test_flattened'

# Ensure processed data directories exist
os.makedirs(PROCESSED_TRAIN_FLATTENED_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_FLATTENED_DIR, exist_ok=True)

# Step 1: Load the dataset
df = pd.read_csv(RAW_DATA_PATH)

# Step 2: Handling missing data (if any)
def handle_missing_data(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)  # Impute mode for categorical
        else:
            df[column].fillna(df[column].median(), inplace=True)  # Impute median for numerical
    return df

df = handle_missing_data(df)

# Step 3: Encoding categorical features
def encode_categorical_data(df):
    label_encoder = LabelEncoder()
    categorical_cols = ['source', 'has_punycode', 'has_internal_links', 'whois_data']  # Example categorical features
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df

df = encode_categorical_data(df)

# Step 4: Outlier detection and removal
def remove_outliers(df, feature_columns):
    z_scores = np.abs(stats.zscore(df[feature_columns]))
    df_cleaned = df[(z_scores < 3).all(axis=1)]  # Keep rows without outliers (Z-score threshold = 3)
    return df_cleaned

feature_cols = ['url_length', 'url_entropy', 'digit_letter_ratio', 'domain_age_days']  # Features for outlier detection
df = remove_outliers(df, feature_cols)

# Step 5: Normalization/Standardization
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Step 6: Drop non-numerical columns like 'url'
df = df.drop(['url'], axis=1)

# Step 7: Train-Test Split
X = df.drop('label', axis=1)  # Replace 'label' with actual target column name
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: PCA for Flattened Data
def apply_pca(data, n_components=50):
    max_components = min(n_components, data.shape[1])  # Ensure n_components is <= number of features
    pca = PCA(n_components=max_components)
    return pca.fit_transform(data)

X_train_flattened = apply_pca(X_train)
X_test_flattened = apply_pca(X_test)


# Step 9: Save the flattened data
np.save(os.path.join(PROCESSED_TRAIN_FLATTENED_DIR, 'X_train_flattened.npy'), X_train_flattened)
np.save(os.path.join(PROCESSED_TEST_FLATTENED_DIR, 'X_test_flattened.npy'), X_test_flattened)
np.save(os.path.join(PROCESSED_TRAIN_FLATTENED_DIR, 'y_train.npy'), y_train.values)
np.save(os.path.join(PROCESSED_TEST_FLATTENED_DIR, 'y_test.npy'), y_test.values)

print("Flattened data preprocessing complete!")
