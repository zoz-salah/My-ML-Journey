# Capstone Project Model 3: Customer Rating Prediction 

**Students:** Ziad Salah , Ziad tamer , Rana Ali , Rawan Samir , Shahd Abdallah

**Track:** Data Science Machine Learning

**Chosen Model:** Model 3 — Rating Prediction (Regression)
## Dataset

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), licensed CC BY-NC-SA 4.0. All raw CSV files used are included under `data/raw/` so the notebook runs without any extra download step.

## Folder structure

```
capstone_project/
├── data/
│   ├── raw/                 8 original Olist CSV files
│   └── clean_orders.csv     cleaned, feature-engineered, order-level dataset (95,829 rows)
├── notebook/
│   └── Capstone_Model3_RatingPrediction_Analysis.ipynb   full 17-section analysis, already executed
├── model/
│   └── best_model.pkl       tuned Random Forest, load with joblib.load()
├── visuals/
│   └── 01-11 numbered PNG charts, referenced throughout the notebook
├── reports/
│   ├── model_comparison.csv     4-algorithm comparison table + tuned result
│   ├── feature_importance.csv   full feature importance list
│   └── error_by_category.csv    prediction error broken down by product category
└── README.md
```

## How to run

Open `notebook/Capstone_Model3_RatingPrediction_Analysis.ipynb` in Jupyter and run all cells (`Restart & Run All`). It reads directly from `../data/raw/`, so keep the folder structure above intact. Full run takes roughly 5-6 minutes (most of it is the hyperparameter tuning step).

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`.

## Headline result

Tuned Random Forest: **MAE ≈ 0.87, RMSE ≈ 1.13, R² ≈ 0.21** on the held-out test set — beating Linear Regression, KNN, and an untuned Decision Tree (which badly overfit, test R² = -0.65).

**Biggest driver of rating by far:** whether the order arrived late relative to its estimated delivery date (63% of feature importance; average rating drops from 4.29 to 2.27 stars for late orders). Full findings are in Section 14 of the notebook.

## Shared team base

Sections 3-9 (Imports → Preprocessing → Feature Engineering → EDA → Train/Test Split & Encoding) form the reusable team foundation — any teammate building one of the other 4 models (Return / Delay / Segmentation / Revenue) can start from the same cleaned `data/clean_orders.csv` and adapt the target/feature selection from Section 9 onward.
