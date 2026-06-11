# 🏥 Medical Insurance Cost Analysis and Prediction System

A production-grade, end-to-end data science project built as a Jupyter Notebook. It takes a local `insurance.csv` dataset and walks through the full ML pipeline — from raw data ingestion through interactive dashboards — covering regression, classification, and visual analytics.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Notebook Structure](#notebook-structure)
- [Models & Techniques](#models--techniques)
- [Key Findings](#key-findings)
- [Output & Visualizations](#output--visualizations)

---

## Project Overview

This notebook provides a complete analytical framework for understanding and predicting medical insurance premium costs. It combines exploratory data analysis, feature engineering, regression modeling, binary classification, and an interactive Plotly dashboard — all in a single, self-contained workflow.

**Use cases:**
- Insurance premium cost forecasting
- High-cost customer risk classification
- Actuarial data exploration and pattern discovery

---

## Dataset

The notebook expects a CSV file named `insurance.csv` with the following columns:

| Column | Type | Description |
|---|---|---|
| `age` | Integer | Age of the primary beneficiary |
| `sex` | String | Gender (`male` / `female`) |
| `bmi` | Float | Body Mass Index (kg/m²) |
| `children` | Integer | Number of dependents covered |
| `smoker` | String | Smoking status (`yes` / `no`) |
| `region` | String | Residential region in the US |
| `charges` | Float | Individual medical costs billed (target variable) |

A commonly used version of this dataset is publicly available on [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance).

---

## Requirements

### Python Libraries

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### Environment

The notebook is designed for **Google Colab** (with Google Drive mounting) but is compatible with any Jupyter environment. See the [Setup](#setup--usage) section for details.

---

## Setup & Usage

### Option A — Google Colab (Recommended)

1. Upload `Medical_Insurance_Cost_Analysis_System_v3.ipynb` to Google Colab.
2. Upload `insurance.csv` to your Google Drive.
3. Mount your Google Drive when prompted in **Cell 1**.
4. In the **configuration cell**, update the `FILE_PATH` variable:

```python
FILE_PATH = "/content/drive/MyDrive/your_folder/insurance.csv"
```

5. Run all cells (`Runtime → Run all`).

### Option B — Local Jupyter

1. Clone or download the notebook.
2. Place `insurance.csv` in an accessible directory.
3. Update `FILE_PATH` to your local path:

```python
FILE_PATH = "/path/to/your/insurance.csv"
```

4. Launch Jupyter and run the notebook.

---

## Notebook Structure

The notebook is organized into 14 sequential sections:

| Section | Title | Description |
|---|---|---|
| 1 | Environment Configuration & Data Loading | Library imports, Google Drive mount, CSV ingestion |
| 2 | Data Cleaning & Preprocessing | Missing value check, duplicate removal, structural review |
| 3 | Descriptive Statistics | Average charges, age, BMI, and dependents summary |
| 4–6 | Categorical Strata & Cost Influence | Group-level cost averages by smoker status, gender, age group, BMI category |
| 7 | Static EDA Visualizations | Seaborn/Matplotlib histograms, scatter plots, box plots, pie charts |
| 8 | Risk Tier Segmentation | Quantile-based macro tier labeling (Low / Medium / High Cost) |
| — | Label Encoding | Categorical-to-numeric transformation using `LabelEncoder` |
| — | Correlation Heatmap | Multi-feature Pearson correlation matrix |
| 9 | Simple Linear Regression | Single-feature model using `age` only |
| 10 | Multiple Linear Regression | Multi-feature model using `age`, `bmi`, `children`, `smoker` |
| 11 | Logistic Regression Classifier | Binary classification: Low Cost (0) vs High Cost (1) |
| 12 | Performance Diagnostics | Classification report, confusion matrix, regression comparison table |
| 13 | Interactive Plotly Dashboard | 8-panel dynamic dashboard with scatter, histogram, bar, box, and heatmap plots |
| 14 | Insights & Conclusions | Summary of strategic findings |

---

## Models & Techniques

### Regression

| Model | Features | Metrics Reported |
|---|---|---|
| Simple Linear Regression | `age` only | R², MAE, MSE, RMSE |
| Multiple Linear Regression | `age`, `bmi`, `children`, `smoker` | R², MAE, MSE, RMSE |

### Classification

| Model | Task | Threshold | Metrics Reported |
|---|---|---|---|
| Logistic Regression | High vs Low cost | Median charges | Accuracy, Precision, Recall, F1, Confusion Matrix |

### Preprocessing

- `LabelEncoder` for categorical features (`sex`, `smoker`, `region`)
- `StandardScaler` for logistic regression input scaling
- `pd.cut` / `pd.qcut` for engineered features (`Age_Group`, `BMI_Category`, `Risk_Tier`)

---

## Key Findings

1. **Smoking is the dominant cost driver.** Smokers incur significantly higher premiums across all age groups and BMI ranges — the single most impactful feature in the dataset.

2. **Multiple features improve regression substantially.** Adding `bmi`, `children`, and `smoker` alongside `age` yields a large reduction in MAE/RMSE and a meaningful increase in R² compared to the single-feature baseline.

3. **Logistic classifier provides reliable risk segmentation.** The binary classifier trained on the four key features achieves strong accuracy and F1 scores, making it viable for automated high-premium profile flagging.

---

## Output & Visualizations

Running the notebook produces:

- **Printed statistics** — averages, group-wise breakdowns, model metrics, and classification reports
- **Static plots** — charges distribution, age/BMI vs charges scatter, smoker box plots, demographic pie charts, correlation heatmap (via Matplotlib/Seaborn)
- **Interactive dashboard** — an 8-panel Plotly dashboard titled *"🏥 MEDICAL INSURANCE DATA SCIENCE SYSTEM DYNAMIC DASHBOARD"*, rendered inline in the notebook

---

## File Structure

```
.
├── Medical_Insurance_Cost_Analysis_System_v3.ipynb   # Main notebook
├── insurance.csv                                      # Input dataset (user-provided)
└── README.md                                          # This file
```
## Visualization

<img width="993" height="626" alt="image" src="https://github.com/user-attachments/assets/05912039-4265-4e8c-9669-7eba18a7b191" />
<img width="573" height="501" alt="image" src="https://github.com/user-attachments/assets/980d622a-a7c5-4a3f-bd75-9883a00a10da" />
<img width="439" height="381" alt="image" src="https://github.com/user-attachments/assets/2342212a-a8ba-48da-914c-ec2ce1dd8c7d" />
<img width="692" height="1003" alt="image" src="https://github.com/user-attachments/assets/6e9a4459-2fae-4ab7-9797-6afffa0b2eea" />


---

## License

This project is for educational and analytical purposes. The insurance dataset used is publicly available and free to use for non-commercial data science work.
