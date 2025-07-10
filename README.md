# Heart Disease Classification using Machine Learning

A supervised learning project that builds and compares multiple machine learning models to predict heart disease based on clinical data. The dataset is sourced from the **UCI Heart Disease Repository** and includes medical attributes like chest pain type, blood pressure, cholesterol levels, and ECG results.

## Problem Definition

Can we use machine learning to predict whether a patient has heart disease based on their clinical parameters?

## Dataset

- **Original Source**: [UCI Heart Disease Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Kaggle Version**: [Heart Disease Classification Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- **Rows**: ~303 samples  
- **Target**: `target` (1 = Disease, 0 = No Disease)

## Exploratory Data Analysis (EDA)

Key insights:
- Balanced class distribution.
- Correlation matrix and bar plots used to explore feature relationships.
- Gender, chest pain type, and max heart rate showed significant influence.

## Features

| Feature     | Description |
|-------------|-------------|
| age         | Age in years |
| sex         | 0 = female, 1 = male |
| cp          | Chest pain type (0–3) |
| trestbps    | Resting blood pressure |
| chol        | Serum cholesterol in mg/dl |
| fbs         | Fasting blood sugar > 120 mg/dl |
| restecg     | Resting electrocardiographic results |
| thalach     | Maximum heart rate achieved |
| exang       | Exercise-induced angina |
| oldpeak     | ST depression |
| slope       | Slope of ST segment |
| ca          | Number of major vessels colored by fluoroscopy |
| thal        | Thalassemia result |
| target      | 1 = Disease, 0 = No Disease |


## Models Used

### Classical Models
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Neural Network

## Methodology

1. **EDA**: Correlations, visualizations, and missing value checks.
2. **Feature Engineering**: Normalization and encoding handled.
3. **Model Training**: Split into 80% training and 20% test.
4. **Model Comparison**: Evaluated using accuracy and classification reports.
5. **Cross-Validation**: 5-fold cross-validation for robust metrics.
6. **Hyperparameter Tuning**:
   - Logistic Regression: `GridSearchCV` + `RandomizedSearchCV`
   - Random Forest: `RandomizedSearchCV`
7. **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Confusion Matrix
   - ROC Curve

## Results

### Model Accuracy (Initial Evaluation)

| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression | ~95% |
| KNN                | ~75% (after tuning) |
| Random Forest      | ~95% |
| SVM                | Not supported for multi-class in current config |
| Neural Network     | ~98% |

### 🔍 Final Cross-Validated Metrics (Logistic Regression)
| Metric     | Score |
|------------|-------|
| Accuracy   | **95%** |
| Precision  | **93%** |
| Recall     | **96%** |
| F1 Score   | **94%** |

---

## Visualizations

- ROC Curve
<Figure size 640x480 with 1 Axes><img width="567" height="432" alt="image" src="https://github.com/user-attachments/assets/414986bc-6bd8-4b49-b4d2-5f0e65cbe396" />
  
- Confusion Matrix
<Figure size 300x300 with 1 Axes><img width="314" height="312" alt="image" src="https://github.com/user-attachments/assets/a7db3160-8766-4c50-9933-ea517a1f3775" />

- Feature Importance (Logistic Regression Coefficients)
<Figure size 640x480 with 1 Axes><img width="587" height="510" alt="image" src="https://github.com/user-attachments/assets/66053eb7-cbaa-4649-a9b5-3a5e3878ec36" />

## Feature Importance (Logistic Regression)

Features with the highest impact on heart disease prediction:

- **thalach (max heart rate)** → Positive influence  
- **cp (chest pain type)** → Positive influence  
- **sex** → Negative influence  
- **oldpeak** → Negative influence  

