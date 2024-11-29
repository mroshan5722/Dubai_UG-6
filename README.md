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
     | Feature 1 | Feature 2 | Output |
     |-----------|-----------|--------|
     | Example 1 | Example 2 | Output |
     | Example 3 | Example 4 | Output |
   - **Steps Taken**: [If any steps were taken to expand or clean the dataset, mention them here. If none, write "No additional steps taken."]

2. **Dataset 2 - CHess Openings**  
   - **Source**:  https://www.kaggle.com/datasets/datasnaek/chess
   - **License**: CC0 1.0 Universal, Wieczerzak, D., Czarnul, P. (2023). Dataset Related Experimental Investigation of Chess Position Evaluation Using a Deep Neural Network. 
   - **Examples**:  
     | Feature 1 | Feature 2 | Output |
     |-----------|-----------|--------|
     | Example 1 | Example 2 | Output |
     | Example 3 | Example 4 | Output |
   - **Steps Taken**: [Mention steps if any. Otherwise, say "No additional steps taken."]

---

### Running the Data Preparation Pipeline <br>

How to run your data preparation pipeline, with a short overview of the steps it includes

A short description (100 words max.) of each requirements (R2–R5) and their location(s) within your repository using permanent links. This includes:
a. What is your model predicting? (a.k.a., the outputs)
b. What is your model using to predict from? (a.k.a., the inputs)
c. A table of results (especially for R4/R5), using Markdown tables
d. A figure showing the results, using permanent links

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

#### Note : Information you will present in your Project Pitch must also be on the GitHub README. <br><br>

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
