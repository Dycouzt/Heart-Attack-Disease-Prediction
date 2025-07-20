# Heart Disease Prediction Using Logistic Regression, XGBoost, and Random Forest

This project analyzes and predicts the likelihood of heart disease based on medical attributes using machine learning models. Originally designed to study Logistic Regression, it evolved to include more powerful algorithms like XGBoost and Random Forest when the dataset's non-linear nature limited the logistic model's effectiveness.

The model is trained and evaluated using the Heart Attack Prediction Dataset from Kaggle, focusing on comparative analysis, performance tuning, and real-world medical prediction.

---

## Goal

The goal of this project was to understand and implement Logistic Regression for a classification task. However, due to the dataset's non-linear relationships, the project pivoted to explore more advanced tree-based models that could better capture the underlying patterns. The final objective became evaluating and comparing multiple models for effective heart disease prediction.

---

## Features

- Loads and cleans the Heart Attack Prediction Dataset from Kaggle
- Implements Logistic Regression as a baseline model
- Introduces XGBoost and Random Forest to handle non-linear features
- Evaluates model performance with accuracy, precision, recall, F1-score, and confusion matrix
- Visualizes feature importance and correlation heatmaps
- Includes preprocessing steps like normalization and label encoding
- Provides insights into which features impact cardiovascular risk

---

## Requirements

This project uses the following Python libraries:
- pandas (for data handling)
- numpy (for numerical operations)
- scikit-learn (for model training and evaluation)
- xgboost (for XGBoost classifier)
- matplotlib and seaborn (for data visualization)

Install the required libraries with:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```plaintext
  HeartAttackDiseasePrediction/
├── logisticregressionprime.ipynb
├── randomforest-ha.ipynb
├── xgboost-ha.ipynb
├── Documentation.pdf
└── README.md 
```      

---

## Dataset

heart-attack-prediction-dataset.csv

---

## How It Works

1. Data Loading:
   - Imports the Heart Attack Prediction Dataset from Kaggle.
   - Inspects data types, null values, and distributions.

2. Preprocessing:
   - Encodes categorical features and normalizes numerical ones.
   - Splits data into train/test sets using stratification.

3. Model Training:
   - Trains Logistic Regression for baseline results.
   - Adds XGBoost and Random Forest to handle complex feature interactions.

4. Evaluation:
   - Calculates accuracy, precision, recall, and F1 score.
   - Plots confusion matrices and ROC curves.
   - Compares performance across models.

5. Interpretation:
   - Displays feature importance scores.
   - Analyzes misclassifications and model limitations.

---

## Code Highlights

- `sklearn.linear_model.LogisticRegression`: Used for initial baseline model.
- `xgboost.XGBClassifier`: Handles non-linear relationships efficiently.
- `sklearn.ensemble.RandomForestClassifier`: Provides robust tree-based classification.
- `sklearn.metrics`: Used for evaluation and confusion matrix.
- `seaborn.heatmap()`: For correlation and feature importance visualizations.

---

## Sample Output

```plaintext
Model Evaluation Results:

Logistic Regression:
Accuracy: 84.1%
F1 Score: 0.81

Random Forest:
Accuracy: 89.7%
F1 Score: 0.88

XGBoost:
Accuracy: 91.2%
F1 Score: 0.90
```

---

## Conclusion

This project began as an exploration of Logistic Regression for predicting heart disease. However, upon discovering the dataset’s complexity and non-linear structure, the scope expanded to include XGBoost and Random Forest — two models better suited for this classification task. The result is a comparative study that not only provides accurate predictions but also highlights the importance of model selection based on data characteristics.

---

## Author

Diego Acosta - Dycouzt
