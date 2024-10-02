import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from sklearn.decomposition import PCA # type: ignore
from scipy import stats # type: ignore
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # type: ignore


# File paths for the raw data and processed data directories
RAW_DATA_PATH = './data/raw_data/phishing/out.csv'
PROCESSED_TRAIN_DIR = './data/processed_data/phishing/train'
PROCESSED_TEST_DIR = './data/processed_data/phishing/test'

# Ensure processed data directories exist
os.makedirs(PROCESSED_TRAIN_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

### Step 1: Load the dataset ###
def load_data(file_path):
    return pd.read_csv(file_path)

### Step 2: Handle missing data ###
def handle_missing_data(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)  # Impute mode for categorical
        else:
            df[column].fillna(df[column].median(), inplace=True)  # Impute median for numerical
    return df

### Step 3: Encoding categorical features ###
def encode_categorical_data(df):
     # Apply LabelEncoder for binary features
    binary_cols = ['label', 'starts_with_ip', 'has_punycode', 'has_internal_links', 'domain_has_digits']
    label_encoder = LabelEncoder()
    for col in binary_cols:
        df[col] = label_encoder.fit_transform(df[col])
    # Apply OneHotEncoder for nominal features
    nominal_cols = ['source']  # whois_data is more complex and may require separate handling
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # 'sparse_output' replaces 'sparse'
    one_hot_encoded = pd.DataFrame(one_hot_encoder.fit_transform(df[nominal_cols]), columns=one_hot_encoder.get_feature_names_out())
    # Drop original nominal columns and concatenate one-hot encoded columns
    df = df.drop(nominal_cols, axis=1)
    df = pd.concat([df, one_hot_encoded], axis=1, ignore_index=False)
    return df


### Step 4: Outlier detection and removal ###
def remove_outliers(df, feature_columns, z_threshold=2.5):
    z_scores = np.abs(stats.zscore(df[feature_columns]))
    df_cleaned = df[(z_scores < z_threshold).all(axis=1)]  # Keep rows without outliers
    return df_cleaned

### Step 5: Scaling features ###
def scale_features(df, feature_columns):
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df

### Step 6: PCA for Flattened Data (optional) ###
def apply_pca(data, n_components=50, use_pca=False):
    if use_pca:
        # Select only numeric columns for PCA
        numeric_data = data.select_dtypes(include=[np.number])
        # Ensure n_components is less than or equal to the number of numeric features
        max_components = min(n_components, numeric_data.shape[1])
        pca = PCA(n_components=max_components)
        data = pca.fit_transform(numeric_data)
    return data


### Step 7: Train-Test Split ###
def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

### Step 8: Save Processed Data ###
def save_data(X_train, X_test, y_train, y_test, train_dir, test_dir, data_type='unflattened'):
    np.save(os.path.join(train_dir, f'X_train_{data_type}.npy'), X_train)
    np.save(os.path.join(train_dir, f'y_train_{data_type}.npy'), y_train)
    np.save(os.path.join(test_dir, f'X_test_{data_type}.npy'), X_test)
    np.save(os.path.join(test_dir, f'y_test_{data_type}.npy'), y_test)

### General Preprocessing Pipeline ###
def preprocessing_pipeline(file_path, use_tokenization=False, use_pca=False, n_components=50):
    # Step 1: Load data
    df = load_data(file_path)
    
    # Step 2: Handle missing data
    df = handle_missing_data(df)
    
    # Step 3: Encode categorical features
    categorical_cols = ['source', 'has_punycode', 'has_internal_links']  # Add more as needed
    df = encode_categorical_data(df)
    
    # Step 4: Remove outliers
    feature_cols = ['url_length', 'url_entropy', 'digit_letter_ratio', 'domain_age_days']
    df = remove_outliers(df, feature_cols)
    
    # Step 5: Scale features
    df = scale_features(df, feature_cols)
    
    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = split_data(df, target_column='label')
    
    # Step 7: Apply PCA (if required)
    X_train = apply_pca(X_train, n_components=n_components, use_pca=use_pca)
    X_test = apply_pca(X_test, n_components=n_components, use_pca=use_pca)
    
    # Step 8: Save processed data
    data_type = 'flattened' if use_pca else 'unflattened'
    save_data(X_train, X_test, y_train, y_test, PROCESSED_TRAIN_DIR, PROCESSED_TEST_DIR, data_type=data_type)
    
    print(f"Data preprocessing complete! Data saved as {data_type}.")

# Run the pipeline
preprocessing_pipeline(RAW_DATA_PATH, use_pca=True, n_components=50)
