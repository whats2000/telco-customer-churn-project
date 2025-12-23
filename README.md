# Telco Customer Churn Analysis

A machine learning project for predicting customer churn using three different models: Spark MLlib, Scikit-Learn, and Keras/TensorFlow.

## Overview

This project builds and compares three churn prediction models on the Telco Customer Churn dataset. It demonstrates data loading, cleaning, exploratory data analysis (EDA), consistent train-test splitting, feature engineering, model training, evaluation, and comparison. The goal is to identify a strong, reliable approach for churn prediction and understand key drivers for retention planning.

## Dataset

The dataset is sourced from Kaggle: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data).

The dataset file is `WA_Fn-UseC_-Telco-Customer-Churn.csv` located in the `data/` directory.

It includes the following key fields:

- `customerID`: Unique identifier for customers (not used as a feature)
- `Churn`: Target variable (Yes/No)
- Numeric features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- Categorical features: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

Note: `TotalCharges` requires cleaning as it arrives as text with blank strings for missing values.

## Features

- Data loading and auditing
- Data cleaning (handling `TotalCharges` blanks)
- Exploratory Data Analysis (EDA) with visualizations
- Stratified 80/20 train-test split
- Feature preprocessing pipelines for each framework
- Three models:
  - Spark MLlib Logistic Regression
  - Scikit-Learn Gradient Boosting or Random Forest
  - Keras/TensorFlow Neural Network
- Model evaluation with ROC-AUC, PR-AUC, F1, precision, recall
- Cross-model comparison and recommendations

## Installation

### Prerequisites

- Python 3.10 or higher
- uv (for dependency management)
- Java (for Spark)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/telco-customer-churn-project.git
   cd telco-customer-churn-project
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   uv run python --version
   ```

## Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   uv run jupyter notebook
   ```

2. Open `telco_churn_analysis.ipynb` and run the cells sequentially.

### Key Steps in the Notebook

1. **Setup**: Initialize libraries and Spark session
2. **Data Loading**: Load and audit the dataset
3. **Data Cleaning**: Handle `TotalCharges` and encode target
4. **EDA**: Analyze churn rates, distributions, and relationships
5. **Train-Test Split**: Stratified 80/20 split
6. **Feature Engineering**: Define and preprocess features
7. **Model 1 - Spark MLlib**: Train and evaluate Logistic Regression
8. **Model 2 - Scikit-Learn**: Train and evaluate tree-based model
9. **Model 3 - Keras**: Train and evaluate neural network
10. **Comparison**: Compare metrics and recommend a model

### Direct Execution

You can also run the notebook cells directly using nbconvert:

```bash
uv run jupyter nbconvert --to notebook --execute telco_churn_analysis.ipynb
```

## Dependencies

- `ipykernel`: Jupyter kernel for Python
- `matplotlib`: Plotting and visualization
- `notebook`: Jupyter Notebook
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `pyspark`: Apache Spark for MLlib
- `scikit-learn`: Machine learning library
- `tensorflow[and-cuda]`: Deep learning framework for Keras

## Model Performance

The project evaluates models using ROC-AUC, PR-AUC, F1-score, precision, and recall. Due to class imbalance, PR-AUC is emphasized. The final recommendation is based on performance, interpretability, and operational complexity.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for the Telco Customer Churn dataset
- The open-source community for the machine learning libraries used
