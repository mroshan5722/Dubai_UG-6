import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Define directories for the datasets
DATA_DIR = "./data/raw_data"
CHESS_DIR = os.path.join(DATA_DIR, "chess")
PHISHING_DIR = os.path.join(DATA_DIR, "phishing")
CLOUDCAST_DIR = os.path.join(DATA_DIR, "cloudcast")

# Ensure directories exist
os.makedirs(CHESS_DIR, exist_ok=True)
os.makedirs(PHISHING_DIR, exist_ok=True)
os.makedirs(CLOUDCAST_DIR, exist_ok=True)

# Function to download datasets from Kaggle
def download_dataset(dataset, path):
    api = KaggleApi()
    api.authenticate()  # Ensure you have your Kaggle API token set up
    print(f"Downloading {dataset} dataset...")
    api.dataset_download_files(dataset, path=path, unzip=True)
    print(f"Downloaded and extracted to {path}")

if __name__ == "__main__":
    # Chess Dataset (small enough to include in repo)
    chess_dataset = 'datasnaek/chess'
    download_dataset(chess_dataset, CHESS_DIR)
    
    # Phishing URL Detection Dataset (large, download needed)
    phishing_dataset = 'sergioagudelo/phishing-url-detection'
    download_dataset(phishing_dataset, PHISHING_DIR)

    # Fashion MNIST Dataset (large, download needed)
    cloudcast_dataset = 'christianlillelund/the-cloudcast-dataset-small'
    download_dataset(cloudcast_dataset, CLOUDCAST_DIR)
    
    print("All datasets downloaded and ready.")