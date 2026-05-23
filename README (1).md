# 🧠 Customer Personality Analysis

A machine learning project that analyzes customer personality traits and predicts their likelihood of responding to marketing campaigns using a Random Forest classifier.

---

## 📌 Overview

This project uses the [Customer Personality Analysis dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) from Kaggle to build a binary classification model that predicts whether a customer will respond (`Response = 1`) to a marketing campaign.

The pipeline covers the full ML workflow: data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and feature importance analysis.

---

## 📂 Project Structure

```
customer-personality-analysis/
│
├── customer_personality_analysis.py   # Main script (full pipeline)
├── cleaned_df.pkl                     # Serialized cleaned DataFrame
├── train_test_data.pkl                # Serialized train/test splits
└── README.md
```

---

## 📊 Dataset

- **Source**: [Kaggle – Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **Format**: Tab-separated CSV (`marketing_campaign.csv`)
- **Target variable**: `Response` (0 = No, 1 = Yes)

Key feature groups include:

| Category | Features |
|----------|----------|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Purchase behavior | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Campaign history | `AcceptedCmp1–5`, `NumDealsPurchases` |
| Enrollment | `Dt_Customer` (parsed into `Customer_Year`, `Customer_Month`, `Customer_Day`, `Customer_Since_Days`) |

---

## ⚙️ Methodology

### 1. Data Cleaning
- Parsed `Dt_Customer` into date components and computed customer tenure in days
- Filled missing `Income` values with the **median**
- Removed outliers using the **IQR method** on all numerical columns
- Dropped irrelevant columns: `ID`, `Z_CostContact`, `Z_Revenue`

### 2. Exploratory Data Analysis (EDA)
- Distribution plots for numerical and categorical features
- Box plots of numerical features vs. `Response`
- Count plots of categorical features vs. `Response`

### 3. Preprocessing Pipeline (`sklearn`)
- **Numerical features**: Median imputation → Standard scaling
- **Categorical features**: One-hot encoding (with `handle_unknown='ignore'`)
- Built using `ColumnTransformer` + `Pipeline` for reproducibility

### 4. Model
- **Algorithm**: `RandomForestClassifier`
- **Handling class imbalance**: `class_weight='balanced'`
- **Train/Test split**: 80/20 with stratification

### 5. Evaluation
- Accuracy score
- Classification report (Precision, Recall, F1)
- Confusion matrix (visualized as heatmap)
- Feature importance plot

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
```

### Run the Script

```bash
python customer_personality_analysis.py
```

> **Note**: The script uses `kagglehub` to automatically download the dataset. Make sure your Kaggle API credentials are configured. See [Kaggle API docs](https://www.kaggle.com/docs/api) for setup.

---

## 📈 Results

The Random Forest model with balanced class weights produces strong classification performance across both classes. Key metrics:

- **Accuracy**: Evaluated on the held-out test set
- **Feature Importance**: Top predictors include income, spending on meat/wine products, and customer tenure

> Detailed metrics are printed at runtime and visualized via matplotlib/seaborn plots.

---

## 💾 Serialized Outputs

After training, the following files are saved with `pickle`:

| File | Contents |
|------|----------|
| `cleaned_df.pkl` | Cleaned and feature-engineered DataFrame |
| `train_test_data.pkl` | `X_train`, `X_test`, `y_train`, `y_test` splits |

These can be loaded for downstream use or model serving without rerunning the pipeline.

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas**, **numpy** – Data manipulation
- **scikit-learn** – Preprocessing, modeling, evaluation
- **matplotlib**, **seaborn** – Visualization
- **kagglehub** – Dataset download
- **pickle** – Model/data serialization

---

## 📝 License

This project is for educational and portfolio purposes. Dataset credit goes to [imakash3011 on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).
