# 🏠 Housing Price Prediction - Holberton Baku ML

This repository contains a machine learning solution for the **Housing Price Prediction** challenge. The project utilizes a **Hybrid Ensemble approach**, combining Gradient Boosting (XGBoost) and Regularized Linear Models (ElasticNet/Ridge) to achieve high predictive accuracy.

## 📊 Project Overview

The goal of this project is to predict the final sale price of homes based on 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa.

**Key Performance Metric:**
- **Validation Score (MAE):** ~$14,807
- **Evaluation Metric:** Mean Absolute Error (on Log-transformed Target)

## 🛠️ Tech Stack

- **Python 3.x**
- **Libraries:**
  - `pandas`, `numpy` (Data Manipulation)
  - `scikit-learn` (Preprocessing, Linear Models, Validation)
  - `xgboost` (Gradient Boosting)

## 🧠 Methodology

To achieve the best results, two distinct modeling approaches were used and then blended:

### 1. Feature Engineering
- **Missing Values:** Imputed using domain knowledge (e.g., missing "Garage" means No Garage, filled with 0).
- **Ordinal Encoding:** Quality features (Ex, Gd, TA...) mapped to numerical scales (5, 4, 3...).
- **New Features:** - `TotalSF` (Total Square Footage)
  - `HouseAge` (Years since built)
  - `TotalBath` (Combined bathroom count)
- **Target Transformation:** `np.log1p()` used to normalize the skewed `SalePrice` distribution.

### 2. Model 1: XGBoost (Tree-based) 
- Handles non-linear relationships and interactions automatically.
- **Configuration:** Low learning rate (0.01), 3000 estimators, and depth constraints to prevent overfitting.

### 3. Model 2: ElasticNet & Ridge (Linear-based) 

[Image of linear regression slope]

- Captures linear trends and extrapolates well.
- **Specific Preprocessing:** - Added **Polynomial Features** (Squared and Cubic terms for `OverallQual`, `TotalSF`) to capture curvature.
  - Applied `RobustScaler` to handle outliers.

### 4. Ensemble (Blending)
The final prediction is a weighted average of the two models to reduce variance:
\[ P_{final} = 0.60 \times P_{Linear} + 0.40 \times P_{XGBoost} \]

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn xgboost
