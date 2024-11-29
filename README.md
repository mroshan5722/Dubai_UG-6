# Project Title <br>

### Group Name: **Dubai Group 6**

### Group Members:
- Hemansi Bhalani
- Ishana Jabbar
- Mehweesh Kadegaonkar
- Muhammed Roshan Palayamkot
- Shakib Moolur

---

### Project Milestones: <br>

| Milestone  | Expected Completion Date |
|------------|--------------------------|
| Requirement 1 (R1) – Data Collection & Setup  | 20/09/24 |
| Requirement 2 (R2) – Exploratory Data Analysis | 27/09/24 |
| Requirement 3 (R3) – Initial Modeling | 10/10/24 |
| Requirement 4 (R4) – Model Evaluation | 20/10/24 |
| Requirement 5 (R5) – Final Report & Presentation | 20/11/24 |

---

### Dataset Sources <br>

1. **Dataset 1 - URL Phishing**  
   - **Source**:  https://www.kaggle.com/datasets/sergioagudelo/phishing-url-detection  
   - **License**: Massachusetts Institute of Technology,Dataset compiled and published in September 2024, Only used in  kaggle notebooks  
   - **Examples**:  

| url                                       | source          | label       | url_length | starts_with_ip | url_entropy | has_punycode | digit_letter_ratio | dot_count | at_count | dash_count | tld_count | domain_has_digits | subdomain_count | nan_char_entropy | has_internal_links | whois_data                                               | domain_age_days |
|-------------------------------------------|-----------------|-------------|------------|----------------|-------------|--------------|--------------------|-----------|----------|------------|-----------|--------------------|------------------|------------------|--------------------|---------------------------------------------------------|-----------------|
| apaceast.cloudguest.central.arubanetworks.com | Cisco-Umbrella | legitimate  | 45         | False          | 3.924535    | False        | 0.0                | 4         | 0        | 0          | 0         | False              | 3                | 0.310387         | False              | {'domain_name': ['ARUBANETWORKS.COM', 'arubane...      | 8250.0          |
| quintadonoval.com                         | Majestic        | legitimate  | 17         | False          | 3.572469    | False        | 0.0                | 1         | 0        | 0          | 0         | False              | 0                | 0.240439         | False              | {'domain_name': ['QUINTADONOVAL.COM', 'quintad...      | 10106.0         |
| nomadfactory.com                          | Majestic        | legitimate  | 16         | False          | 3.327820    | False        | 0.0                | 1         | 0        | 0          | 0         | False              | 0                | 0.250000         | False              | {'domain_name': ['NOMADFACTORY.COM', 'nomadfac...      | 8111.0          |
| tvarenasport.com                          | Majestic        | legitimate  | 16         | False          | 3.500000    | False        | 0.0                | 1         | 0        | 0          | 0         | False              | 0                | 0.250000         | False              | {'domain_name': ['TVARENASPORT.COM', 'tvarenas...      | 5542.0          |
| widget.cluster.groovehq.com               | Cisco-Umbrella | legitimate  | 27         | False          | 3.930270    | False        | 0.0                | 3         | 0        | 0          | 0         | False              | 2                | 0.352214         | False              | {'domain_name': 'GROOVEHQ.COM', 'registrar': '...      | 5098.0          |

   - **Steps Taken**:
  

   - **Dropped Missing Values:** Removed rows or columns with missing data to ensure consistency and avoid errors during training.
   
   - **Created Two Subsets:** Divided the data into two subsets for focused processing and analysis.
   
   - **Shuffled the Data:** Randomized the order of samples.
   
   - **Saved Cleaned Subsets:** Saved the processed subsets to use them for the different model implementations.
   
   - **Split into Train-Test Sets:** Divided the data into training and testing sets to evaluate model performance effectively.
   
   - **Model-Specific Steps:** Additional preprocessing steps were applied in the same file as required for the individual model implementations.

2. **Dataset 2 - Chess Openings**  
   - **Source**:  https://www.kaggle.com/datasets/datasnaek/chess
   - **License**: CC0 1.0 Universal, Wieczerzak, D., Czarnul, P. (2023). Dataset Related Experimental Investigation of Chess Position Evaluation Using a Deep Neural Network. 
   - **Examples**:  
     

   | id         | rated   | created_at   | last_move_at | turns | victory_status | winner | increment_code | white_id     | white_rating | black_id     | black_rating | moves                                                                                          | opening_eco | opening_name                     | opening_ply |
   |------------|---------|--------------|--------------|-------|-----------------|--------|----------------|--------------|--------------|--------------|--------------|------------------------------------------------------------------------------------------------|-------------|-----------------------------------|-------------|
   | TZJHLljE   | False   | 1.504210e+12 | 1.504210e+12 | 13    | outoftime       | white  | 15+2           | bourgris     | 1500         | a-00         | 1191         | d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5...                                                 | D10         | Slav Defense: Exchange Variation  | 5           |
   | l1NXvwaE   | True    | 1.504130e+12 | 1.504130e+12 | 16    | resign          | black  | 5+10           | a-00         | 1322         | skinnerua    | 1261         | d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6...                                               | B00         | Nimzowitsch Defense: Kennedy Variation | 4           |
   | mIICvQHh   | True    | 1.504130e+12 | 1.504130e+12 | 61    | mate            | white  | 5+10           | ischia       | 1496         | a-00         | 1500         | e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc...                                               | C20         | King's Pawn Game: Leonardis Variation | 3           |
   | kWKvrqYL   | True    | 1.504110e+12 | 1.504110e+12 | 61    | mate            | white  | 20+0           | daniamurashov| 1439         | adivanov2009 | 1454         | d4 d5 Nf3 Bf5 Nc3 Nf6 Bf4 Ng4 e3 Nc6 Be2 Qd7 O...                                               | D02         | Queen's Pawn Game: Zukertort Variation | 3           |
   | 9tXo1AUZ   | True    | 1.504030e+12 | 1.504030e+12 | 95    | mate            | white  | 30+3           | nik221107    | 1523         | adivanov2009 | 1469         | e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 N...                                               | C41         | Philidor Defense                  | 5           |

   - **Steps Taken**: The following preprocessing steps were performed on the dataset:
   -    **Removing Irrelevant Columns**: Unnecessary columns were discarded to retain only the relevant data for analysis and model training.
   -    **Label-Encoding Categorical Features**: Categorical variables were encoded into numerical values to make them compatible with machine learning models.
   -    **Defining the Target Variable**: The target variable, representing the outcome of the match, was identified for prediction purposes.
   -    **Splitting the Dataset**: The dataset was divided into training and testing sets to evaluate the model's performance effectively.

3. **Dataset 3 - Traffic Sign**  
   - **Source**:  https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification
   - **License**: -
   - **Examples**:  
     | Feature 1 | Feature 2 | Output |
     |-----------|-----------|--------|
     | Example 1 | Example 2 | Output |
     | Example 3 | Example 4 | Output |
   - **Steps Taken**: Had to balance this dataset

---

### Running the Data Preparation Pipeline <br>

1. **Set up the Virtual Environment**:
   - First, download and set up the virtual environment to ensure all necessary dependencies are isolated for the project.

2. **Download Datasets**:
   - Use the provided `download_data.py` script to download the datasets directly from Kaggle. Make sure you have your Kaggle API credentials configured before running the script.

3. **Run Preprocessing Scripts**:
   - Execute the relevant preprocessing scripts for each dataset to clean, transform, and prepare the data for model training. This step includes tasks such as removing irrelevant columns, encoding categorical features, and splitting the dataset into training and testing sets.

4. **Run the Jupyter Notebook**:
   - Once the datasets are preprocessed, run the Jupyter Notebook for the selected model to perform the training and evaluation. The notebook will guide you through the steps of model building, evaluation, and results interpretation.

---

### Files and Folders Structure <br>
| File/Folder | Purpose |
|-------------|---------|
| data/	| Contains raw and processed datasets |
| scripts/ | Contains data pipeline and preprocessing scripts |
| notebooks/ | Contains Jupyter notebooks for EDA and modeling |
| requirements.txt | List of Python dependencies for running the project |
| README.md | Project overview and instructions |

<br><br>
---

## Setting Up a Virtual Environment and Installing Dependencies

To ensure consistency across all environments, please set up a virtual environment and install the required dependencies using the `requirements.txt` file. Follow the steps below depending on your operating system. <br>

---

### For Mac and Linux Users:

1. **Create a Virtual Environment**:
   In the terminal, run the following command to create a virtual environment:
   ```bash
   python3 -m venv f20dl
   ```
2. **Activate the Virtual Environment**: 
After creating the virtual environment, activate it:
```bash
source f20dl/bin/activate
```
3. **Install the Dependencies**: 
With the virtual environment activated, install the project dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
4. **Deactivate the Virtual Environment**: 
Once you're done working, deactivate the virtual environment by running:
```bash
deactivate
```
<br>

---

### For Windows Users:

1. **Create a Virtual Environment**: 
In Command Prompt or PowerShell, run the following command to create a virtual environment:
```bash
python -m venv f20dl
```
2. **Activate the Virtual Environment**: 
After creating the virtual environment, activate it:
```bash
.\f20dl\Scripts\activate
```
3. **Install the Dependencies**: 
With the virtual environment activated, install the project dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
4. **Deactivate the Virtual Environment**:
Once you're done working, deactivate the virtual environment by running:
```bash
deactivate
```
<br>

### Additional Notes:
- Ensure you have Python 3.6+ installed on your system before running the commands.
- It’s important to always activate the virtual environment before running the project and deactivate it when done to avoid issues with global Python packages.
<br><br>

---

## Data Pipeline Setup and Instructions

This project uses the **Kaggle API** to automatically download, extract, and preprocess datasets. Follow the instructions below to set up the Kaggle API and run the data pipeline on **Mac**, **Windows**, and **Linux** systems. <br>

### Prerequisites:
1. **Python** installed on your system (Python 3.6+).
2. **Kaggle API** credentials (`kaggle.json` file).
3. Required Python libraries (listed in `requirements.txt`).<br>

---

### Step 1: Set Up Kaggle API Credentials
To use the Kaggle API, you need to set up your API credentials (i.e., the `kaggle.json` file).

#### 1.1 Download your Kaggle API Key:
1. Go to [Kaggle](https://www.kaggle.com/) and sign in.
2. Navigate to your account settings by clicking on your profile icon and selecting "Account".
3. Scroll down to the **API** section and click on **Create New API Token**. This will download a file named `kaggle.json`.<br>

---

### Step 2: Install the Kaggle API

Install the Kaggle API, which is available as a Python package:

```bash
pip install kaggle
```
<br>

---

### Step 3: Configure the Kaggle API

Next, place the kaggle.json file in the correct location depending on your operating system. <br>


#### 3.1 For Mac and Linux Users:

Move the kaggle.json file to the .kaggle directory in your home folder:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
```
Set the correct permissions for the kaggle.json file:

```bash
chmod 600 ~/.kaggle/kaggle.json
```
<br>

#### 3.2 For Windows Users:

Move the `kaggle.json` file to the `.kaggle` directory in your user folder:  

1. Open `File Explorer`.  
2. Navigate to `C:\Users\<Your-Username>\`.  
3. Create a folder named `.kaggle` if it doesn’t exist.  
4. Move `kaggle.json` into `C:\Users\<Your-Username>\.kaggle\`.  

<br>

---

### Step 4: Running the Pipeline Script

Once the Kaggle API is set up, you can run the data pipeline script to automatically download, extract, and preprocess the datasets. <br>

#### 4.1 Run the Pipeline

In the terminal, navigate to the directory where the project is located and run the following command:

```bash
python3 scripts/download_data.py
```
<br>

This script will:

1. Download the datasets from Kaggle.  
2. Extract the datasets (if compressed).  
3. Preprocess the datasets and save them to the appropriate folder.  

The datasets will be organized as follows:

- Raw datasets will be saved in the `data/raw_data/` folder.  
- Preprocessed datasets will be saved in the `data/processed_data/` folder.  <br>

### Troubleshooting:

Error: `kaggle: command not found`: Ensure the Kaggle API is installed by running pip install kaggle.
Error: Permission denied for `kaggle.json`: Ensure the correct permissions are set for `kaggle.json`:

For Mac/Linux: `chmod 600 ~/.kaggle/kaggle.json`
For Windows: Make sure the file is placed in the correct directory and is not write-protected. <br>
<br>

---

### Summary of Commands:
**For Mac/Linux:**
```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
python scripts/download_data.py
```
<br>

**For Windows:**
```bash
pip install kaggle
# Move kaggle.json to C:\Users\<Your-Username>\.kaggle\
python scripts/download_data.py
```
---
