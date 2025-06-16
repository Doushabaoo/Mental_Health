# Mental Health Risk Prediction Project
**Kaggle Competition** (https://www.kaggle.com/competitions/playground-series-s4e11/overview)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.2.2-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-green)

A machine learning pipeline to predict depression risk factors.

## Key Features
- **Data Pipeline**: Automated preprocessing for:
  - Missing value imputation
  - Feature engineering (Pressure/Satisfaction ratio)
  - One-hot encoding
- **Model Development**: 
  - XGBoost classifier (94.1% accuracy)
  - Hyperparameter tuning with RandomizedSearchCV
