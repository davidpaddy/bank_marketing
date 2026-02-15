# Machine Learning Assignment 2  
## Bank Marketing Classification using Multiple ML Models

---

## 1. Problem Statement
The objective of this project is to build and compare multiple machine learning
classification models to predict whether a customer will subscribe to a term
deposit based on demographic and marketing campaign information.

The project demonstrates an end-to-end ML workflow including:
- Data preprocessing
- Model training and evaluation
- Model comparison
- Streamlit-based deployment

---

## 2. Dataset Description
- Dataset: Bank Marketing Dataset (Kaggle)
- Type: Binary Classification
- Instances: ~11,000 records
- Features: Customer demographics, financial attributes, and campaign details
- Target Variable: `deposit`
  - yes → customer subscribed
  - no → customer did not subscribe

### Features Used
age, job, marital, education, default, balance, housing, loan, contact,
day, month, duration, campaign, pdays, previous, poutcome

---

## 3. Machine Learning Models Implemented
The following models were trained using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## 4. Evaluation Metrics
Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression | 0.8258 | 0.9072 | 0.8270 | 0.7996 | 0.8131 | 0.6504 |
| Decision Tree | 0.7922 | 0.7912 | 0.7856 | 0.7722 | 0.7788 | 0.5830 |
| KNN | 0.7783 | 0.8416 | 0.7998 | 0.7098 | 0.7521 | 0.5561 |
| Naive Bayes | 0.7331 | 0.8227 | 0.7992 | 0.5832 | 0.6743 | 0.4738 |
| Random Forest | 0.8625 | 0.9192 | 0.8326 | 0.8885 | 0.8596 | 0.7267 |
| XGBoost | 0.8643 | 0.9257 | 0.8410 | 0.8800 | 0.8600 | 0.7292 |

---

## 6. Observations

| ML Model | Observation about Model Performance |
|---------|-------------------------------------|
| Logistic Regression | Provides a strong baseline performance with high AUC, indicating that several relationships in the dataset are approximately linear. Shows stable and interpretable results. |
| Decision Tree | Captures nonlinear decision boundaries but exhibits slightly lower generalization performance due to overfitting tendencies. |
| KNN | Performance is moderate and sensitive to feature scaling and neighborhood size. Computationally heavier during prediction compared to other models. |
| Naive Bayes | Fast and computationally efficient but lower recall and MCC indicate limitations due to the feature independence assumption. |
| Random Forest | Demonstrates significant improvement over single trees by reducing variance through ensemble averaging, resulting in strong recall and balanced performance. |
| XGBoost | Achieves the best overall performance with highest AUC and MCC, benefiting from gradient boosting, regularization, and sequential learning of residual errors. |

---

## 7. Streamlit Application
The deployed Streamlit application provides:

- CSV dataset upload option
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report

### Live App Link
(Add Streamlit URL here)

---

## 8. Repository Structure
