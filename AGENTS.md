# Telco Customer Churn — 3-Model Benchmark (Spark MLlib + Scikit-Learn + Keras)

## Goal
Build and compare **three different churn prediction models** using a consistent **80/20 train-test split** (stratified on `Churn`) on the Telco Customer Churn dataset. The objective is to identify a strong, reliable approach and understand key churn drivers for retention planning. :contentReference[oaicite:0]{index=0}

---

## Overview of Steps
Data Loading → Data Audit/Cleaning → EDA → Single Stratified 80/20 Split → Feature Engineering Pipelines → Train 3 Models (Spark / Sklearn / Keras) → Evaluate & Compare → Report Insights

---

## Data Structure and Submission Format
### Dataset file
- Source file path to load in the notebook: `/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv` :contentReference[oaicite:1]{index=1}

### Target
- `Churn` (categorical): **Yes/No** (convert to binary for modeling)

### Identifier
- `customerID` (string): unique per customer; **do not use as a feature** (keep only for joins/splitting consistency)

### Feature fields (recommended types)
- Numeric:
  - `SeniorCitizen` (0/1 integer)
  - `tenure` (integer months)
  - `MonthlyCharges` (float)
  - `TotalCharges` (should be numeric; arrives as text and needs cleaning)
- Categorical (Yes/No or multi-class):
  - `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`,
    `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`,
    `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

### Notes that must be handled
- `TotalCharges` contains **blank strings** in a small number of rows; treat blanks as missing and resolve during cleaning.
- No sample submission is provided; **no `submission.csv` is required**.

---

# Step 1. Notebook Setup & Data Loading

### Objective
Initialize libraries (Spark + pandas + sklearn + TensorFlow/Keras), load the dataset, and establish reproducible settings.

### Instructions
1. Set a single global random seed to reuse across pandas/sklearn/Spark/TensorFlow.
2. Start a local Spark session configured for MLlib.
3. Load the CSV into a pandas DataFrame for auditing/EDA and keep the raw copy unchanged.
4. Confirm row count, column names, and basic dtypes.

### Requirements / Tools
- Spark (PySpark MLlib)
- pandas
- scikit-learn
- TensorFlow/Keras (for deep learning)
- A single fixed `random_state`/seed used everywhere

### Expected Output
✅ `df_raw` — unmodified pandas DataFrame  
✅ `spark` — active Spark session  

---

# Step 2. Data Audit & Cleaning (Schema-Faithful)

### Objective
Produce a cleaned dataset with correct numeric types and consistent target encoding.

### Instructions
1. Verify `customerID` uniqueness; confirm there are no duplicates.
2. Clean `TotalCharges`:
   - Strip whitespace.
   - Convert blank strings to missing.
   - Convert to numeric type.
3. Decide and apply a missing-value policy for `TotalCharges`:
   - Preferred: impute with a business-consistent rule (e.g., if `tenure == 0` then `TotalCharges = 0`; otherwise impute with median or with `MonthlyCharges * tenure` if appropriate), and document the choice.
4. Normalize target:
   - Map `Churn`: Yes→1, No→0 (keep the original label column if you want interpretability).
5. Ensure all categorical columns are treated as categorical/string consistently (important for encoders later).

### Requirements / Tools
- pandas for cleaning and checks
- Clear documentation of the imputation strategy

### Expected Output
✅ `df_clean` — cleaned pandas DataFrame with numeric `TotalCharges` and binary churn label  
✅ A short markdown note describing the `TotalCharges` handling rule  

---

# Step 3. Exploratory Data Analysis (EDA)

### Objective
Understand churn prevalence, key relationships, and feature distributions to guide modeling and evaluation choices.

### Instructions
1. Report churn rate and class imbalance (counts + percentage).
2. Summarize numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) overall and split by churn.
3. Summarize categorical feature churn rates (e.g., churn by `Contract`, `InternetService`, `PaymentMethod`).
4. Visualizations to include (keep readable and minimal):
   - Target distribution bar chart
   - Numeric feature distributions split by churn
   - Top categorical churn-rate comparisons (a few most informative)
5. Note any leakage risks (none expected) and any odd values (e.g., tenure=0 patterns).

### Requirements / Tools
- matplotlib (or Spark/pandas plotting)
- Keep EDA lightweight; the goal is insight + sanity checks

### Expected Output
✅ EDA section with a few plots + bullet insights on likely churn drivers  

---

# Step 4. One Consistent 80/20 Train-Test Split (Stratified)

### Objective
Create a **single** 80/20 split that is reused across Spark, scikit-learn, and Keras for fair comparison.

### Instructions
1. Perform a stratified 80/20 split in pandas using `Churn` as the stratification column.
2. Preserve `customerID` in both splits so you can:
   - Reconstruct Spark DataFrames that match the same rows
   - Keep evaluation consistent
3. Create:
   - `df_train` (80%)
   - `df_test` (20%)
4. Confirm churn rate is similar in both splits (sanity check).

### Requirements / Tools
- scikit-learn split utilities (conceptually; implement in notebook)
- Use the same seed as in Step 1

### Expected Output
✅ `df_train`, `df_test` — consistent train/test datasets  
✅ A short check showing train/test churn rates are comparable  

---

# Step 5. Shared Feature Definition & Preprocessing Strategy

### Objective
Define the feature set once and implement equivalent preprocessing in each modeling framework.

### Instructions
1. Define:
   - `ID_COL = customerID`
   - `TARGET_COL = churn_binary` (or equivalent)
   - Numeric features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
   - Categorical features: all remaining non-target, non-ID columns
2. Preprocessing rules:
   - Numeric: impute if needed (should be minimal after cleaning), scale/standardize for models that benefit (logistic regression, neural net).
   - Categorical: one-hot encode (handle unknown categories in test).
3. Keep a consistent “feature contract” documented in markdown so results are interpretable and reproducible.

### Requirements / Tools
- Spark: StringIndexer + OneHotEncoder + VectorAssembler (+ optional StandardScaler)
- scikit-learn: ColumnTransformer + OneHotEncoder (+ optional StandardScaler)
- Keras: reuse the same encoded/scaled arrays as sklearn, or implement equivalent preprocessing layers

### Expected Output
✅ A markdown “Feature Contract” listing numeric/categorical features and preprocessing rules  

---

# Step 6. Model 1 — Spark MLlib (Baseline Linear Model)

### Objective
Train a strong, interpretable Spark pipeline model for churn prediction.

### Instructions
1. Convert `df_train`/`df_test` pandas → Spark DataFrames (ensuring the exact same rows via `customerID` membership).
2. Build a Spark ML pipeline:
   - StringIndex categorical columns
   - OneHotEncode indexed categoricals
   - Assemble all features into `features`
   - (Optional) scale assembled features if using a linear model
3. Train a **Spark Logistic Regression** classifier (recommended baseline in Spark).
4. Generate predictions and predicted probabilities on the Spark test set.
5. Evaluate with:
   - ROC-AUC
   - PR-AUC (helpful for imbalance)
   - Confusion matrix metrics at a chosen threshold (default 0.5, plus optionally tune threshold based on business preference)

### Requirements / Tools
- `pyspark.ml` Pipeline components
- Spark evaluators for AUC metrics

### Expected Output
✅ `spark_lr_model` — fitted Spark pipeline  
✅ Spark test metrics table (AUC + precision/recall/F1 + confusion matrix)  

---

# Step 7. Model 2 — Scikit-Learn (Nonlinear Tree-Based)

### Objective
Train a strong nonlinear sklearn model that can capture interactions and nonlinearity.

### Instructions
1. Build a sklearn preprocessing pipeline:
   - ColumnTransformer for numeric + categorical
   - OneHotEncoder for categorical (handle unknown categories)
   - Scaling for numeric if the chosen model benefits (tree models usually do not require scaling)
2. Choose one primary tree-based model (pick one and keep it consistent):
   - Recommended: Gradient boosting family available in sklearn (e.g., HistGradientBoostingClassifier) OR RandomForestClassifier as a simpler baseline.
3. Fit on `df_train` and predict on `df_test`.
4. Evaluate using the same metrics as Spark:
   - ROC-AUC, PR-AUC
   - F1, precision, recall
   - Confusion matrix
5. Add basic interpretability:
   - Feature importance (for tree models) or permutation importance on the test set
   - Identify top drivers (keep it high-level; one-hot features can be numerous)

### Requirements / Tools
- scikit-learn pipelines + metrics
- Prefer a model that provides probability estimates

### Expected Output
✅ `sk_model` — trained sklearn pipeline  
✅ `sk_metrics` — metrics summary + a small “top drivers” section  

---

# Step 8. Model 3 — Deep Learning (Keras/TensorFlow MLP)

### Objective
Train a compact neural network model and compare performance against classical approaches.

### Instructions
1. Use a consistent encoded representation:
   - Preferred: reuse the sklearn preprocessing output to produce dense/sparse matrices, then convert to a neural-net-friendly format (document how you convert and manage sparsity).
2. Define an MLP architecture suitable for tabular data:
   - A few Dense layers with dropout and/or batch normalization
   - Output layer: sigmoid (binary classification)
3. Training protocol:
   - Train on training split with an internal validation split (from train only) or use a small stratified validation subset from `df_train`
   - Early stopping on validation AUC (or validation loss) to avoid overfitting
4. Evaluate on the untouched `df_test`:
   - ROC-AUC, PR-AUC
   - Threshold-based metrics and confusion matrix
5. Record training curves (loss and AUC) for reporting.

### Requirements / Tools
- TensorFlow/Keras
- Early stopping + fixed seed for reproducibility

### Expected Output
✅ `keras_model` — trained neural net  
✅ Training curves plot + final test metrics  

---

# Step 9. Cross-Model Evaluation & Comparison

### Objective
Make results comparable and select a best candidate for retention use-cases.

### Instructions
1. Build a single comparison table across the 3 models:
   - ROC-AUC, PR-AUC, F1, precision, recall, accuracy
2. Compare confusion matrices side-by-side at:
   - Default threshold 0.5
   - (Optional) a tuned threshold that maximizes F1 or prioritizes recall (if retention outreach is the goal)
3. Add lightweight calibration checks (optional but valuable):
   - Reliability curve or compare predicted probability distributions for churn vs non-churn
4. Summarize tradeoffs:
   - Interpretability (Spark LR best)
   - Raw performance (often boosting / NN)
   - Operational complexity (Spark vs sklearn vs TF)

### Requirements / Tools
- Consistent metric computation across frameworks
- Keep plots readable; don’t overwhelm with too many charts

### Expected Output
✅ `model_comparison_table` — one table to rule them all  
✅ A short written conclusion naming a recommended model + why  

---

# Reporting / Visualization (Optional)

### What to include
- Final model comparison table
- Key churn drivers (top 5–10) with short explanations
- A concise “What we learned” section:
  - Who churns most
  - Which services/contracts are churn-risky
  - How the model could support retention actions

✅ Output: `final_report_section` (markdown cells + plots)

---

# Test / Deployment (Optional)

### Instructions
- Define a “single customer scoring” flow:
  - Required input fields
  - Cleaning rules (especially `TotalCharges`)
  - Preprocessing reuse (pipeline reuse is critical)
- Show how to generate churn probability and map to an action threshold (e.g., outreach list).

✅ Output: `scoring_flow_description`

---

# Technical Notes
- **Language:** Python (single Jupyter Notebook)
- **Libraries:** pandas, numpy, matplotlib, pyspark (MLlib), scikit-learn, tensorflow/keras
- **Reproducibility:** Use one global seed and reuse it in pandas split, Spark, sklearn, and TensorFlow
- **Evaluation:** Prefer ROC-AUC + PR-AUC due to class imbalance; always include threshold metrics for business interpretability
- **Data caveat:** `TotalCharges` must be cleaned/coerced to numeric and missing resolved before modeling

---

# Deliverables
- One notebook (end-to-end): load → clean → EDA → split → 3 models → evaluation → conclusion
- Metrics artifacts:
  - Per-model metrics + confusion matrix
  - Cross-model comparison table
- Visual artifacts:
  - EDA plots
  - (Optional) training curves for Keras
- A final short recommendation paragraph describing which model to use and why
