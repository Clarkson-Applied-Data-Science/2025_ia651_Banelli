# Credit Card Fraud Detection using Machine Learning

**Author:** Mallikarjun Reddy Banelli  
**Supervisor:** Professor Michael Gilbert  
**University:** Clarkson University

---

## INTRODUCTION

Credit card fraud poses a significant threat in today’s digital economy. As online transactions surge, fraudsters employ sophisticated tactics to exploit payment system vulnerabilities. Timely detection and prevention are essential for financial security and customer trust.

This project leverages machine learning to detect fraudulent credit card transactions using a Kaggle dataset. The goal is to build accurate classifiers capable of identifying fraud in a highly imbalanced dataset, where fraudulent transactions constitute only 0.172% of records. Emphasis is placed on **Recall**, **F1 Score**, and **ROC-AUC** to evaluate model performance beyond accuracy.

---

## OBJECTIVE

To develop and evaluate multiple machine learning models that accurately detect fraudulent credit card transactions, minimizing false negatives and enhancing generalization through feature engineering, data balancing (SMOTE), and hyperparameter tuning.

---

## DATASET

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Shape:** 284,807 rows × 31 columns

### 📊 Class Distribution

| Class | Description      | Count   |
|-------|------------------|---------|
| 0     | Non-Fraudulent   | 284,315 |
| 1     | Fraudulent       | 492     |

**Features:**

- `Time`, `Amount`  
- `V1` to `V28`: Anonymized principal components via PCA  
- `Class`: Target (0 = Non-Fraud, 1 = Fraud)

---

## PROCESS OVERVIEW

- **Exploratory Data Analysis (EDA):** Visualized class imbalance, feature correlations, and fraud indicators  
- **Data Preprocessing:** Applied `StandardScaler` and performed train-test split  
- **Balancing Strategy:** Used SMOTE to address class imbalance in the training set  
- **Modeling:** Trained five classification models  
- **Hyperparameter Tuning:** Used `HalvingRandomSearchCV` and `RandomizedSearchCV`  
- **Evaluation:** Confusion matrices, F1 Score, Recall, Precision, and ROC-AUC on both unbalanced and balanced test sets

 ### 🧪 After SMOTE – Balanced Class Distribution

  | Class | Description      | Count    |
  |-------|------------------|----------|
  | 0     | Non-Fraudulent   | 227,451  |
  | 1     | Fraudulent       | 227,451  |

---

## MODELS IMPLEMENTED

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Machine (LinearSVC with HalvingRandomSearchCV)  
- XGBoost Classifier

---

## EVALUATION METRICS

To ensure effective fraud detection, models were evaluated using:

- **Recall**: Critical for detecting fraud cases  
- **Precision**: To minimize false alarms  
- **F1 Score**: Balances Recall and Precision  
- **ROC-AUC**: Measures ability to distinguish between classes  
- **Confusion Matrix**: Visualizes false positives and false negatives

---

## RESULTS & COMPARISON

Models were evaluated on both the **original unbalanced test set** (20% of data) and a **SMOTE-balanced test set**.  
Random Forest and XGBoost achieved the highest performance, particularly in Recall and F1 Score.


| Model               | Accuracy | Recall  | Precision | F1 Score | ROC AUC |
|--------------------|----------|---------|-----------|----------|---------|
| SVC                | 99.46%   | 83.67%  | 22.10%    | 34.97%   | 91.58%  |
| Random Forest      | 99.82%   | 89.80%  | 48.62%    | 63.08%   | 94.82%  |
| Logistic Regression| 97.41%   | 91.84%  | 5.79%     | 10.89%   | 94.63%  |
| Decision Tree      | 99.76%   | 78.57%  | 39.49%    | 52.56%   | 89.18%  |
| XGBoost            | 99.66%   | 89.80%  | 32.23%    | 47.44%   | 94.74%  |


---

##  BEST MODEL

**Random Forest Classifier**

The Random Forest model achieved the best overall balance of performance across key metrics:
- **Highest Accuracy:** 99.82%
- **Strong Recall:** 89.80% (important for catching most fraud cases)
- **Best F1 Score:** 63.08% (harmonizes Precision and Recall)
- **Highest ROC-AUC:** 94.82%

While other models like XGBoost and Logistic Regression performed well on individual metrics, Random Forest delivered the most consistent and reliable results across all evaluation criteria, making it the most suitable model for this fraud detection task.

---
