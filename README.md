
# 🏡 Airbnb Price Prediction

## 🚀 Overview
This repository contains a complete machine learning pipeline to predict Airbnb listing prices using structured features. It includes thorough exploratory data analysis (EDA), preprocessing, neural network modeling, ensemble learning, and advanced meta-modeling techniques.

---

## 📊 Dataset Summary
The dataset includes:
- Listing details: `id`, `name`, `rating`, `reviews`
- Host info: `host_name`, `host_id`
- Location: `address`, `country`
- Listing features: `bedrooms`, `bathrooms`, `guests`, `beds`, `studios`, `toilets`
- Amenities, `checkin`, `checkout`
- Target: `price`

---

## 📌 Workflow

### 1. Exploratory Data Analysis (EDA)
- Parsed and separated `features` into usable numerical fields.
- Visualized `price` distribution (found to be right-skewed).
- Identified and treated missing values.
- Visualized relationships between categorical features (`country`) and `price`.

### 2. Data Cleaning & Preprocessing
- Filled missing values (mode/median).
- Applied `log1p()` to normalize skewed columns.
- Scaled numerical features with `RobustScaler`.
- One-hot encoded the `country` column.

### 3. ANN Model Creation
A deep ANN was created using Keras with:
- Layer configuration: 512 → 256 → 128 → 64 → 32 → Output
- Regularization with `Dropout` and `BatchNormalization`
- `Adam` optimizer with learning rate 0.0003
- EarlyStopping + ReduceLROnPlateau

📈 **Deep ANN Performance**:
- MAE: 3730.03
- RMSE: 5063.66
- R² Score: 0.4193

---

## 🔧 Hyperparameter Tuning

### Attempt 1:
- Shallow ANN (256 → 128 → 64) with Dropout
- Learning rate: 0.0005

📉 Performance:
- MAE: 3744.77
- RMSE: 5021.75
- R²: 0.4289

### Attempt 2 (Best):
- Deep ANN with Dropout and BatchNorm
- Learning rate: 0.0003 + scheduler
- EarlyStopping patience: 12

📈 Performance:
- MAE: 3730.03
- RMSE: 5063.66
- R²: 0.4193

---

## 🧪 Accuracy Improvement Techniques

### ✅ Cross-Validation (5-Fold)
- MAE: 3715.96
- RMSE: 5102.25
- R²: 0.4177

### ✅ Tree-Based Models
| Model                  | MAE    | RMSE    | R² Score |
|------------------------|--------|---------|----------|
| Random Forest          | 3830.56| 5145.10 | 0.4005   |
| Gradient Boosting      | 3713.65| 5077.34 | 0.4161   |
| HistGradientBoosting   | 3675.56| 4970.70 | 0.4404   |
| Stacked Regressor      | 3594.24| 4875.47 | 0.4617   |

### ✅ Final Combined Model (Meta-Ensemble)
Combined ANN, Random Forest, Gradient Boosting, and HGB using a Neural Network as a meta-learner.

📈 **Final Performance**:
- MAE: 3527.32
- RMSE: 4850.01
- R²: 0.4673 ✅

> Note: RidgeCV, Linear Regression, and optimized weighted averaging were also tested, but the meta-model neural network gave the best performance.

---

## 📐 Evaluation Metrics

- **MAE (Mean Absolute Error)**: Measures average magnitude of error (lower is better).
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors (lower is better).
- **R² Score (Coefficient of Determination)**: Proportion of variance explained (higher is better).

---

## 🛠️ Tools & Libraries
- Python, pandas, numpy, matplotlib, seaborn
- scikit-learn (RandomForest, GradientBoosting, stacking, metrics)
- TensorFlow / Keras

---

## 📂 Structure
- `Airbnb_Price_Prediction.ipynb`: Main notebook with code and plots
- `Airbnb_Price_Prediction_Explanation.docx`: Detailed project explanation
- `README.md`: This file

---

## ✅ Status
📌 Completed and submitted as part of the SparkTech AI Developer Interview Task.

---
