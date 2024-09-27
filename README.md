Group Name : Dubai Group 6

Project Title :

Group Members :
1. Hemansi Bhalani
2. Ishana Jabbar
3. Mehweesh Kadegaonkar
4. Muhammed Roshan Palayamkot
5. Shakib Moolur
   
Your Project Milestones, including when you are expecting to complete each requirement within context of the semester

The sources of your dataset(s), including:
a. The true source of where the data is from
b. The original license for each source
c. Two specific examples from your datasets, presented nicely
d. Any additional steps taken during data collection to expand the dataset, and any metrics associated with these. If you have not done this, don’t worry.

How to run your data preparation pipeline, with a short overview of the steps it includes

A short description (100 words max.) of each requirements (R2–R5) and their location(s) within your repository using permanent links. This includes:
a. What is your model predicting? (a.k.a., the outputs)
b. What is your model using to predict from? (a.k.a., the inputs)
c. A table of results (especially for R4/R5), using Markdown tables
d. A figure showing the results, using permanent links

Files/folders created in GitHub and their purpose.

# Information you will present in your Project Pitch must also be on the GitHub README.


## Data Pipeline Setup and Instructions

This project uses the **Kaggle API** to automatically download, extract, and preprocess datasets. Follow the instructions below to set up the Kaggle API and run the data pipeline on **Mac**, **Windows**, and **Linux** systems.

### Prerequisites:
1. **Python** installed on your system (Python 3.6+).
2. **Kaggle API** credentials (`kaggle.json` file).
3. Required Python libraries (listed in `requirements.txt`).

---

### Step 1: Set Up Kaggle API Credentials
To use the Kaggle API, you need to set up your API credentials (i.e., the `kaggle.json` file).

#### 1.1 Download your Kaggle API Key:
1. Go to [Kaggle](https://www.kaggle.com/) and sign in.
2. Navigate to your account settings by clicking on your profile icon and selecting "Account".
3. Scroll down to the **API** section and click on **Create New API Token**. This will download a file named `kaggle.json`.

---

### Step 2: Install the Kaggle API

Install the Kaggle API, which is available as a Python package:

```bash
pip install kaggle
```
---

### Step 3: Configure the Kaggle API

Next, place the kaggle.json file in the correct location depending on your operating system.


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


#### 3.2 For Windows Users:

Move the `kaggle.json` file to the `.kaggle` directory in your user folder:  
    1. Open `File Explorer`.  
    2. Navigate to `C:\Users\<Your-Username>\`.  
    3. Create a folder named `.kaggle` if it doesn’t exist.  
    4. Move `kaggle.json` into `C:\Users\<Your-Username>\.kaggle\`.  

---

### Step 4: Install Python Libraries
Ensure you have all the required libraries by installing them from the `requirements.txt` file. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

---

### Step 5: Running the Pipeline Script

Once the Kaggle API is set up, you can run the data pipeline script to automatically download, extract, and preprocess the datasets.

#### 5.1 Run the Pipeline

In the terminal, navigate to the directory where the project is located and run the following command:

```bash
python3 scripts/download_data.py
```

This script will:

1. Download the datasets from Kaggle.  
2. Extract the datasets (if compressed).  
3. Preprocess the datasets and save them to the appropriate folder.  

The datasets will be organized as follows:

- Raw datasets will be saved in the data/raw_data/ folder.  
- Preprocessed datasets will be saved in the data/processed_data/ folder.  

### Troubleshooting:

Error: `kaggle: command not found`: Ensure the Kaggle API is installed by running pip install kaggle.
Error: Permission denied for `kaggle.json`: Ensure the correct permissions are set for `kaggle.json`:

For Mac/Linux: `chmod 600 ~/.kaggle/kaggle.json`
For Windows: Make sure the file is placed in the correct directory and is not write-protected.

### Summary of Commands:
**For Mac/Linux:**
```bash
pip install kaggle
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
pip install -r requirements.txt
python scripts/download_data.py
```

**For Windows:**
```bash
pip install kaggle
# Move kaggle.json to C:\Users\<Your-Username>\.kaggle\
pip install -r requirements.txt
python scripts/download_data.py
```
---