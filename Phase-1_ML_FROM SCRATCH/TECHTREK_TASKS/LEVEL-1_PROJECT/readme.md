# Capstone Project — Model 3: Customer Rating Prediction

## Team

**Students:** Ziad Salah, Ziad Tamer, Rana Ali, Rawan Samir, Shahd Abdallah

**Track:** Data Science & Machine Learning

**Selected Problem:** Model 3 — Customer Rating Prediction

**Problem Type:** Regression

---

## Project Overview

This project focuses on predicting **customer review ratings** for orders using the Brazilian E-Commerce Public Dataset by Olist.

The goal is to build and compare multiple regression models, identify the best-performing model, and analyze the factors that contribute most to customer ratings.

The project covers the complete machine learning workflow:

- Data Understanding
- Data Cleaning & Preparation
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Train/Test Split
- Encoding & Scaling
- Model Training
- Model Comparison
- Hyperparameter Tuning
- Model Evaluation
- Feature Importance
- Final Saving

---

## Dataset

**Brazilian E-Commerce Public Dataset by Olist**

Source: [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

The dataset contains information about orders, customers, products, sellers, payments, and reviews from the Brazilian e-commerce platform Olist.

All required raw CSV files are included in:

```text
data/raw/
````

Therefore, the notebook can be run without downloading the dataset again if you import it from Kaggle, as long as the repository structure is preserved.

---

## Repository Structure

```text
capstone_project/
│
├── data/
│   ├── raw/
│   │   └── 8 original Olist CSV files
│   │
│   └── clean_orders.csv
│       └── Cleaned and feature-engineered order-level dataset
│
├── notebook/
│   └── Capstone_Model3_RatingPrediction_Analysis.ipynb
│       └── Complete 17-section analysis
│
├── model/
│   └── best_model.pkl
│       └── Tuned Random Forest model
│
├── visuals/
│   └── 01-11 numbered PNG charts
│       └── Visualizations used throughout the analysis
│
├── reports/
│   ├── model_comparison.csv
│   │   └── Model comparison and tuned model results
│   │
│   ├── feature_importance.csv
│   │   └── Feature importance results
│   │
│   └── error_by_category.csv
│       └── Prediction error by product category
│
└── README.md
```

---

## Models

Four regression algorithms were trained and compared:

1. **Linear Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Decision Tree Regressor**
4. **Random Forest Regressor**

After comparing the models, **Random Forest** was selected for further hyperparameter tuning using `GridSearchCV`.

---

## Evaluation Metrics

The models were evaluated using:

* **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
* **RMSE (Root Mean Squared Error):** Measures prediction error while giving more weight to larger errors.
* **R² (R-Squared):** Measures how much of the variation in customer ratings is explained by the model.

For MAE and RMSE, lower values are better.
For R², higher values are better.

---

## Results

The tuned Random Forest achieved the following results on the held-out test set:

| Metric | Result |
| ------ | -----: |
| MAE    | ≈ 0.87 |
| RMSE   | ≈ 1.13 |
| R²     | ≈ 0.21 |

The tuned Random Forest performed better than the other tested models.

The untuned Decision Tree showed strong overfitting, with a test **R² of approximately -0.65**, demonstrating why model evaluation and hyperparameter tuning are important.

---

## Key Finding

The most important feature in the final Random Forest model was **whether the order arrived late compared with its estimated delivery date**.

It accounted for approximately **63% of the model's feature importance**.

The analysis also showed a large difference in average ratings:

* **On-time orders:** approximately **4.29 stars**
* **Late orders:** approximately **2.27 stars**

This indicates that delivery performance is strongly associated with customer satisfaction in this dataset.

The complete analysis and supporting visualizations are available in the notebook.

---

## Error Analysis

In addition to the standard evaluation metrics, the project includes error analysis to understand where the model performs poorly.

The analysis investigates:

* Prediction errors by delivery status
* Prediction errors by actual rating
* Errors across product categories
* The worst-performing predictions
* Patterns among large prediction errors

This helps move beyond simply comparing model scores and provides a better understanding of the model's limitations.

---

## How to Run

Open the following notebook:

```text
notebook/Capstone_Model3_RatingPrediction_Analysis.ipynb
```

Run the notebook using:

```text
Restart & Run All
```

The notebook reads the raw data from:

```text
../data/raw/
```

Therefore, the repository folder structure should remain unchanged.

### Requirements

The project uses the following Python libraries:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

You can install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

The complete notebook run takes approximately **5–6 minutes**, with most of the time spent on hyperparameter tuning.

---

## Shared Team Foundation

The preprocessing and analysis developed in **Sections 3–9** serve as the shared foundation for the team's other models.

These sections cover:

* Imports
* Data Loading
* Data Understanding
* Data Cleaning
* Preprocessing
* Feature Engineering
* Exploratory Data Analysis
* Train/Test Split
* Encoding
* Scaling

The resulting:

```text
data/clean_orders.csv
```

can be reused by teammates working on the other capstone models, with the target and feature selection adapted according to each problem.

---

## Project Outcome

This project demonstrates a complete end-to-end regression workflow, from raw e-commerce data to model training, tuning, evaluation, and interpretation.

The final model and analysis can be used as a foundation for further improvements, including additional feature engineering, model optimization, and deployment.

```
```
