import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import logging
from logging_code import setup_logging
logger = setup_logging("feature_scaling")
from sklearn.preprocessing import StandardScaler
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def fs(X_train, y_train, X_test, y_test):
    try:
        logger.info(f"Training data independent size : {X_train.shape}")
        logger.info(f"Training data dependent size : {y_train.shape}")
        logger.info(f"Testing data independent size : {X_test.shape}")
        logger.info(f"Testing data dependent size : {y_test.shape}")
        logger.info(f"Before scaling : {X_train.head(1)}")

        # Standard Scaling
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        # Save the scaler
        with open('standar_scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)
        logger.info(f"After scaling : {X_train_sc[0]}")

        # ----------------------------
        # Hyperparameter tuning for Logistic Regression
        # ----------------------------
        param_grid = {
            'C': np.logspace(-3, 3, 7),      # regularization strength
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # penalty type
            'solver': ['saga'],               # saga supports all penalties
            'max_iter': [100, 200, 500]      # iterations
        }

        reg = LogisticRegression()
        rs = RandomizedSearchCV(
            estimator=reg,
            param_distributions=param_grid,
            n_iter=10,       # number of random combinations to try
            scoring='accuracy',
            cv=5,
            verbose=0,
            n_jobs=-1,
            random_state=42
        )

        rs.fit(X_train_sc, y_train)
        best_model = rs.best_estimator_

        logger.info(f"Best Hyperparameters: {rs.best_params_}")
        logger.info(f"Test Accuracy : {accuracy_score(y_test, best_model.predict(X_test_sc))}")
        logger.info(f"Test Confusion Matrix : \n{confusion_matrix(y_test, best_model.predict(X_test_sc))}")
        logger.info(f"Classification Report : \n{classification_report(y_test, best_model.predict(X_test_sc))}")

        # Save the best model
        with open('Model.pkl', 'wb') as t:
            pickle.dump(best_model, t)

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")