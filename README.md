# Airbnb-Price-Prediction
Airbnb Price Prediction – Technical Report
Objective

The goal of this task was to predict the price of Airbnb listings based on a structured dataset. The project involved exploratory data analysis, preprocessing, modeling with deep neural networks, and ensemble learning techniques. 
Step 1: Exploratory Data Analysis (EDA)

The first step was to inspect the dataset to understand the types of features, identify potential missing values, detect anomalies, and gauge feature distributions.

Key decisions and insights:
- Parsed the `features` column to extract numerical data such as `bedrooms`, `beds`, `guests`, `bathrooms`.
- Visualized the distribution of the `price` variable and found it was heavily right-skewed. Decided to apply a log transform later.
- Checked `rating`, `reviews`, and room counts for extreme outliers.
- Analyzed the relationship between categorical variables (like `country`) and price using boxplots and group means.

Step 2: Data Cleaning & Preprocessing

We needed to ensure the data was clean and suitable for machine learning models.

Steps taken and reasoning:
- Missing values in `checkin`, `checkout`, and `host_name` were filled using the mode since these are likely to be repeated strings.
- Numerical missing values (like `beds`, `guests`, `bathrooms`) were filled using the median to prevent skewing due to outliers.
- Applied `np.log1p()` to skewed features (`reviews`, `guests`, `beds`, etc.) to normalize their distributions and improve learning for ANN.
- Used `RobustScaler` to scale numerical features because it is less sensitive to outliers than `StandardScaler`.
- One-hot encoded the `country` feature to allow categorical learning for both tree-based and neural models.

Step 3: ANN Model Development

The next step was to build a deep learning model using Keras.

Architecture decisions:
- Used 5 dense layers (512 → 256 → 128 → 64 → 32 → 1) to allow the network to capture high-dimensional relationships.
- Applied `BatchNormalization` after each dense layer to stabilize learning and allow higher learning rates.
- Used `Dropout` for regularization to prevent overfitting, especially given limited data.
- Chose `Adam` optimizer with a learning rate of 0.0003 for efficient gradient updates.
- Used `EarlyStopping` to stop training when validation loss didn’t improve and `ReduceLROnPlateau` to dynamically lower learning rate.

Performance:
- MAE: 3730.03
- RMSE: 5063.66
- R² Score: 0.4193

This baseline ANN showed promise but needed further improvements. The evaluation was done with Mean Absolute Error, Root Mean Squared Error, Co-efficient of determination (R square) as this is a Regression Task. 

Step 4: Accuracy Improvement

To improve generalization and performance, multiple strategies were applied.

4.1 Hyper Parameter Tuning
To improve the performance of the base ANN model, I conducted multiple rounds of architectural and training-related hyperparameter tuning. Below are the experiments and the rationale behind each design choice:
Attempt 1:
Decisions & Reasoning:
•	Layer sizes: 256 → 128 → 64 — this is a standard decreasing pattern to compress features progressively.
•	Dropout: Introduced dropout to reduce overfitting, especially on the larger layers.
•	Learning rate: Set to 0.0005, slightly higher than default to allow faster convergence in early epochs.
•	Patience: Early stopping with patience=10 to avoid unnecessary training.
Outcome:
•	Performance improved moderately but plateaued, indicating room for deeper architecture and dynamic learning rate scheduling.
Performance:
-	MAE: 3744.77
-	RMSE: 5021.75
-	R² Score: 0.4289
Attempt 2:
Decisions & Reasoning:
•	Deeper network: Added more layers and units (up to 512) to allow the model to learn more abstract, high-dimensional patterns in the data.
•	Batch Normalization: Placed after each hidden layer to stabilize and speed up training by normalizing activations.
•	Dropout rates: Carefully decreased as depth increased (0.5 → 0.4 → 0.3 → 0.2) to balance regularization and learning capacity.
•	Learning rate: Lowered to 0.0003 to allow fine-tuning in deeper networks.
•	ReduceLROnPlateau: Dynamically reduced learning rate when validation loss plateaued to escape shallow local minima.
•	EarlyStopping: Patience of 12 epochs was used to preserve the best-performing model weights.
Outcome:
•	This configuration outperformed all previous ANN models.
•	It was later chosen as the base ANN to combine with ensemble models in the final stacked architecture.
Deep ANN Performance:
-	MAE: 3730.03
-	RMSE: 5063.66
-	R² Score: 0.4193

4.2 Cross-Validation

5-Fold cross-validation was used to evaluate the model’s generalizability.
- Avoided over-relying on a single train/test split.
- Helped evaluate how consistent the ANN is across data subsets.

Result:
- Average MAE: 3715.96
- Average RMSE: 5102.25
- Average R² Score: 0.4177

4.3 Tree-Based Models

I trained multiple tree-based models using raw + encoded data.

Model decisions and performance:
- Random Forest: Chosen for its ability to handle non-linearities and categorical splits.
  - MAE: 3830.56, RMSE: 5145.10, R²: 0.4005

- Gradient Boosting: Chosen for sequential learning and better bias reduction.
  - MAE: 3713.65, RMSE: 5077.34, R²: 0.4161

- HistGradientBoosting: Selected for fast training and histogram-based splitting.
  - MAE: 3675.56, RMSE: 4970.70, R²: 0.4404

- Stacked Regressor: Combined multiple base learners + MLPRegressor with Ridge as final estimator.
  - MAE: 3594.24, RMSE: 4875.47, R²: 0.4617

4.4 Final Combined Model (ANN + HGB + RF + GB)

To further enhance accuracy, I performed meta-modeling by combining predictions from:
- ANN
- Random Forest
- Gradient Boosting
- HistGradientBoosting

Decision: Use `Neural Network` as a meta-learner to optimally blend predictions (stacked generalization).
This approach was selected because it is interpretable, fast, and often performs well in ensemble tasks.

Final Results:
- MAE: 3527.32
- RMSE: 4850.01
- R² Score: 0.4673 (best achieved)
I tried with Linear Regression, RidgeCV, Optimized weight average as well. But the neural network used as meta-modelling gave best result.


Conclusion

This project reflects a complete machine learning pipeline from EDA to advanced ensemble modeling. Each modeling decision—from log transforms to dropout layers to stacking—was made based on data properties and validation outcomes. The final blended model provided the best performance and balanced bias-variance tradeoff effectively.

Tools & Libraries Used
• Python (pandas, numpy, matplotlib, seaborn)
• Scikit-learn (RandomForest, GradientBoosting, stacking, metrics)
• TensorFlow/Keras (for building ANN)
