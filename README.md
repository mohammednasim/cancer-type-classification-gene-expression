# Cancer Type Classification Using Gene Expression Data

## Project Overview

Cancer Type Classification Using Gene Expression Data is a machine learning project that aims to classify cancer types using gene expression measurements and their corresponding labels.

The project demonstrates a complete machine learning workflow, including data inspection, preprocessing, exploratory data analysis (EDA), feature selection, model training, model comparison, performance evaluation, and prediction through a web interface.

This project is intended for educational and research purposes in machine learning, bioinformatics, and healthcare data analytics.

## Objectives

- Analyze gene expression data and corresponding cancer labels.
- Inspect data quality, missing values, duplicate records, and feature types.
- Perform data preprocessing and feature selection.
- Train and compare multiple machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Provide predictions for new samples through a Python-based web application.

## Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Model Persistence:** Joblib

## Project Structure

```text
Cancer-Type-Classification/
│
├── Dataset/
│   ├── data.csv
│   └── labels.csv
│
├── models/
│   └── best_model.joblib
│
├── results/
│   ├── figures/
│   ├── dataset_audit.json
│   ├── eda_report.md
│   ├── metrics.json
│   └── model_comparison.csv
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── data_utils.py
├── inspect_data.py
├── inspect_dataset.py
├── eda.py
├── train.py
├── evaluate.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The dataset, trained model, and generated results are excluded from the GitHub repository where appropriate. The structure above illustrates the local project layout.*

## Machine Learning Workflow

### 1. Data Collection

The project uses two CSV files:

- `data.csv` — gene expression measurements and associated sample information, depending on the source dataset.
- `labels.csv` — corresponding cancer class labels.

The dataset source, sample alignment, and label definitions should be verified before model training.

### 2. Data Inspection and Preprocessing

- Inspect dataset dimensions, columns, and data types.
- Identify missing values, duplicates, and invalid values.
- Align expression samples with their corresponding labels.
- Prepare numerical features for machine learning.
- Apply preprocessing and feature selection as implemented in the training pipeline.

### 3. Exploratory Data Analysis

Exploratory analysis can include:

- Distribution of cancer classes.
- Gene feature variance.
- Principal Component Analysis (PCA) visualization.
- Dataset quality summaries.

### 4. Model Training and Comparison

The training pipeline is designed to compare the following classification approaches:

- Dummy Classifier — baseline performance.
- Logistic Regression.
- Linear Support Vector Machine (SVM).
- Random Forest.
- Extra Trees Classifier.

The best model is selected according to the configured cross-validation macro-F1 score.

### 5. Model Evaluation

The project supports evaluation using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

The held-out test set is used to assess the selected model's performance.

### 6. Prediction and Web Interface

The prediction module accepts new samples with the expected feature schema and generates class predictions.

A Flask web application provides an interface for uploading a CSV file and viewing the prediction results.

## Installation

### Prerequisites

- Python 3.10 or a compatible Python version.
- Git.
- pip.

### Step 1: Clone the repository

```bash
git clone https://github.com/mohammednasim/cancer-type-classification-gene-expression.git
cd cancer-type-classification-gene-expression
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Dataset Setup

The raw dataset is not included in this repository because of its size.

Obtain the dataset from the original provider or an authorized source. Place the required files in the local `Dataset/` directory:

```text
Dataset/
├── data.csv
└── labels.csv
```

**Important:** Verify that the files have the expected format and that each sample is correctly matched with its label. Do not assume that the row order is correct without checking the dataset documentation.

## Running the Project

Run the following commands from the project root directory, in order:

### 1. Inspect the dataset

```bash
python data_utils.py
python inspect_data.py
```

### 2. Perform exploratory data analysis

```bash
python eda.py
```

### 3. Train and compare models

```bash
python train.py
```

### 4. Evaluate the trained model

```bash
python evaluate.py
```

### 5. Launch the web application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

The trained model and any required local artifacts must be generated before using the prediction interface.

## Results

The project generates evaluation reports, model comparison tables, and visualizations in the local `results/` directory.

Actual performance depends on the dataset, preprocessing, class distribution, and evaluation methodology. Model accuracy and other scores should be reported only after running the complete pipeline and verifying the generated metrics.

## Applications

- Educational bioinformatics projects.
- Gene expression data analysis.
- Machine learning classification experiments.
- Healthcare data analytics research.
- Comparative evaluation of classification algorithms.

## Limitations

- Model performance depends on the quality and representativeness of the dataset.
- High-dimensional gene expression data may contain noise and correlated features.
- Results may vary across datasets and validation strategies.
- A model trained on one dataset may not generalize to other populations or laboratories without further validation.

**Disclaimer:** This project is for educational and research purposes only. It is not a clinically validated diagnostic tool and must not be used to make medical decisions.

## Future Enhancements

- Add dimensionality-reduction methods and additional feature-selection techniques.
- Explore more classification algorithms and hyperparameter tuning.
- Improve the prediction interface with clearer input validation and result visualization.
- Evaluate generalization using independent datasets.
- Investigate explainability methods to understand which features influence predictions.

## Author

**Mohammed Nasim P**

Data Science | Machine Learning | AI | Data Analytics

- GitHub: https://github.com/mohammednasim
- LinkedIn: https://www.linkedin.com/in/mohammednasimpdatascientist/

---

If you find this project useful for learning machine learning and bioinformatics, consider starring the repository.
