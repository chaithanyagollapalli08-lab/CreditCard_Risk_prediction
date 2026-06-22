# 🏦 AI-Powered Credit Risk Prediction System

## 📌 Project Overview

The **AI-Powered Credit Risk Prediction System** is a Machine Learning-based web application that predicts whether a customer is likely to be a **Good Customer** or **Bad Customer** based on financial, demographic, and credit-related information.

The system helps financial institutions assess customer creditworthiness and reduce the risk of Non-Performing Assets (NPA) by making data-driven lending decisions.

---

## 🎯 Problem Statement

Banks and financial institutions often face significant losses due to loan defaults and NPAs. Identifying high-risk customers before approving loans is critical for effective risk management.

This project leverages Machine Learning techniques to analyze customer information and predict potential credit risk, enabling smarter lending decisions.

---

## 🚀 Features

* Data Cleaning and Preprocessing
* Missing Value Handling
* Outlier Detection and Treatment
* Feature Engineering
* Feature Selection using Statistical Methods
* Feature Scaling using StandardScaler
* Machine Learning Model Training
* Credit Risk Prediction
* Interactive Flask Web Application
* Responsive and User-Friendly Interface

---

## 📊 Dataset Features

The model uses the following customer attributes:

| Feature                                  | Description              |
| ---------------------------------------- | ------------------------ |
| Revolving Utilization of Unsecured Lines | Credit utilization ratio |
| Age                                      | Customer age             |
| Monthly Income                           | Monthly earnings         |
| Number of Open Credit Lines and Loans    | Active credit accounts   |
| Number of Real Estate Loans or Lines     | Property-related loans   |
| Number of Dependents                     | Family dependents        |
| Gender                                   | Customer gender          |
| Region                                   | Customer location        |
| Housing Status                           | Rented / Own House       |
| Occupation                               | Employment category      |
| Education                                | Education level          |

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* SciPy
* Pickle

### Web Framework

* Flask

### Frontend

* HTML
* CSS

### Machine Learning

* Logistic Regression
* Feature Selection
* StandardScaler

---

## ⚙️ Machine Learning Workflow

### 1. Data Collection

* Customer credit dataset

### 2. Data Preprocessing

* Missing value treatment
* Duplicate removal
* Outlier handling
* Data transformation

### 3. Feature Engineering

* Encoding categorical variables
* Feature transformation
* Scaling numerical features

### 4. Feature Selection

#### Variance Threshold

Removes low-variance features from the dataset.

#### Hypothesis Testing

Uses Pearson Correlation and P-Value analysis to select statistically significant features.

### 5. Data Scaling

```python
StandardScaler()
```

### 6. Model Training

```python
LogisticRegression()
```

### 7. Model Evaluation

Performance metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🌐 Web Application

The Flask application allows users to:

* Enter customer details
* Submit financial information
* Predict customer risk
* View instant prediction results

### Prediction Output

#### 🟢 Good Customer

Customer has a lower probability of becoming an NPA.

#### 🔴 Bad Customer

Customer has a higher probability of becoming an NPA.

---

## 🔧 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Credit-Risk-Prediction.git
```

### Navigate to Project

```bash
cd AI-Credit-Risk-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Flask Application

```bash
python app.py
```

### Open in Browser

```text
http://127.0.0.1:5000/
```

---

## 📈 Future Enhancements

* XGBoost Implementation
* Random Forest Comparison
* Deep Learning Models
* Explainable AI (SHAP)
* Credit Risk Score Visualization
* Cloud Deployment
* REST API Integration

---

## 💡 Learning Outcomes

Through this project, I gained practical experience in:

* Data Preprocessing
* Feature Engineering
* Feature Selection
* Statistical Analysis
* Machine Learning Model Development
* Model Deployment
* Flask Application Development
* End-to-End ML Pipeline Design

---

## 👨‍💻 Author

**Chaithanya Gollapalli**

AI/ML Enthusiast | Data Science Learner | Machine Learning Developer


