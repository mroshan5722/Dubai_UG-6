import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # type: ignore
from scipy import stats  # type: ignore

# File paths for the raw data and processed data directories
RAW_DATA_PATH = './data/raw_data/phishing/out.csv'
PROCESSED_TRAIN_UNFLATTENED_DIR = './data/processed_data/phishing/train_unflattened'
PROCESSED_TEST_UNFLATTENED_DIR = './data/processed_data/phishing/test_unflattened'

# Ensure processed data directories exist
os.makedirs(PROCESSED_TRAIN_UNFLATTENED_DIR, exist_ok=True)
os.makedirs(PROCESSED_TEST_UNFLATTENED_DIR, exist_ok=True)

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
    categorical_cols = ['url', 'source', 'has_punycode', 'has_internal_links']  # Example categorical features
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df

df = encode_categorical_data(df)

# Step 4: Tokenization (optional, based on raw URLs)
def tokenize_url(df):
    # Example: Use CountVectorizer to tokenize the 'url' column
    vectorizer = CountVectorizer()
    tokenized_urls = vectorizer.fit_transform(df['url'])
    tokenized_df = pd.DataFrame(tokenized_urls.toarray(), columns=vectorizer.get_feature_names_out())
    return pd.concat([df, tokenized_df], axis=1).drop('url', axis=1)

# Uncomment if using raw URLs:
# df = tokenize_url(df)

# Step 5: Outlier detection and removal
def remove_outliers(df, feature_columns):
    z_scores = np.abs(stats.zscore(df[feature_columns]))
    df_cleaned = df[(z_scores < 3).all(axis=1)]  # Keep rows without outliers (Z-score threshold = 3)
    return df_cleaned

feature_cols = ['url_length', 'url_entropy', 'digit_letter_ratio']
df = remove_outliers(df, feature_cols)

# Step 6: Train-Test Split
X = df.drop('label', axis=1)  # Replace 'label' with actual target column name
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Save the unflattened data as numpy arrays
np.save(os.path.join(PROCESSED_TRAIN_UNFLATTENED_DIR, 'X_train_unflattened.npy'), X_train.to_numpy())
np.save(os.path.join(PROCESSED_TEST_UNFLATTENED_DIR, 'X_test_unflattened.npy'), X_test.to_numpy())
np.save(os.path.join(PROCESSED_TRAIN_UNFLATTENED_DIR, 'y_train.npy'), y_train.to_numpy())
np.save(os.path.join(PROCESSED_TEST_UNFLATTENED_DIR, 'y_test.npy'), y_test.to_numpy())

print("Unflattened data preprocessing complete and saved as .npy!")
