# SMS Classifier - Ensemble Learning Web Implementation

This project is the **ensemble-learning part** of a web-based SMS/email classifier.
It includes:

- a cleaned preprocessing pipeline
- two ensemble models: **Soft Voting** and **Stacking**
- automatic comparison between the two models using a validation split
- a proper **train/test split** from the main dataset
- a separate **15-row confidence set** used only after training
- a web app built with **Streamlit**

## Python version

Use **Python 3.13** for this project.

## What the model predicts

The model predicts the **message type**:

- `spam` - suspicious, scam-like, or unsolicited content
- `gov` - government advisory or public-service message
- `notifs` - service or account notification
- `otp` - one-time password or verification message
- `ads` - promotional content

## Dataset design

### Main dataset

Used for training and formal evaluation:

- `data/raw/main_dataset.xlsx`
- `data/raw/main_dataset.csv`

Workflow:
- preprocess the full main dataset
- create an **80/20 train/test split**
- split the training portion again into **train/validation** for model comparison
- use the held-out test split for final evaluation

### Separate confidence set

Used only after training:

- `data/confidence/confidence_set.xlsx`
- `data/confidence/confidence_set.csv`

This is the 15-row file and is **not used for training**.

## Project structure

```
sms-classifier-ensemble-py313-clean/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── confidence/
├── src/
│   ├── utils.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_ensemble.py
│   ├── evaluate_confidence.py
│   └── predict.py
├── models/
└── outputs/
```

## What each file does

### `src/preprocessing.py`
Loads the dataset, normalizes column names, cleans the message text, and builds the processed `clean_text` column.

### `src/train_ensemble.py`
Trains both ensemble methods, compares them on the validation split, retrains the best one on the full training portion, evaluates on the held-out test split, and saves the models plus reports.

### `src/evaluate_confidence.py`
Runs the saved best model on the external 15-row confidence set and saves the predictions.

### `src/predict.py`
Lets you test one message from the terminal.

### `app.py`
Runs the Streamlit website. The user enters a message, clicks **Predict**, and the app shows:
- predicted message type
- label meaning
- confidence per class

## Fresh start setup

### 1. Extract the zip
Unzip the project somewhere easy to access, such as your Downloads folder.

### 2. Open a terminal in the project folder
The correct folder is the one that contains:
- `requirements.txt`
- `app.py`
- `src`
- `data`

### 3. Create a virtual environment

#### Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux/macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run order

### A. Train and compare the ensemble models
```bash
python .\src\train_ensemble.py
```

This creates:
- `models/soft_voting_model.pkl`
- `models/stacking_model.pkl`
- `models/best_model.pkl`
- `outputs/reports/ensemble_comparison.csv`
- `outputs/reports/best_model_test_metrics.json`
- `outputs/reports/best_model_classification_report.txt`

### B. Evaluate the 15-row confidence set
```bash
python .\src\evaluate_confidence.py
```

This creates:
- `outputs/reports/confidence_set_predictions.csv`
- `outputs/reports/confidence_set_classification_report.txt`

### C. Launch the website
```bash
streamlit run .\app.py
```

Then open the local URL shown in the terminal.

## Quick terminal test

```bash
python .\src\predict.py
```

## Website behavior

The web app already has a message input field. The user:
1. pastes or types a message
2. clicks **Predict**
3. sees the predicted message type and confidence scores

## If installation fails

Check your Python version:
```bash
python --version
```

It should say **Python 3.13.x**.
