# Vashist-ML-Hackathon

This repository contains the work I submitted for the **GDSC Vashisht ML Hackathon (Open For All)**. In this hackathon, participants used a football results dataset to predict the outcome of a football match based on various features.

## Overview

The challenge involved using a dataset containing football match data to predict the result of a randomly selected match. The dataset included features such as team names, match date, tournament, city, country, and whether the match was played at a neutral venue. The task was to build a machine learning model that could predict whether the home team won, lost, or if the match ended in a draw.

## Dataset Description

The dataset consists of the following files:

### 1. **Training Set:**
- **X_train.csv**: Features for training the model.
- **y_train.csv**: True labels (match results) corresponding to `X_train.csv`. The results are:
  - **0**: Draw
  - **1**: Loss to the home team
  - **2**: Win for the home team

### 2. **Test Set:**
- **X_test.csv**: Features used for testing the model.

### 3. **Sample Submission:**
- **sample_submission.csv**: Example submission file showing the correct format for predictions.

### 4. **Columns in X:**
- **date**: The date the match was played (YYYY-MM-DD format).
- **home_team**: Name of the home team.
- **away_team**: Name of the away team.
- **tournament**: Name of the tournament.
- **city**: City where the match took place.
- **country**: Country where the match took place.
- **neutral**: Indicates whether the match was played at a neutral venue (True/False).

### 5. **Y Column (Target Variable):**
- **result**: The match outcome:
  - **0** = Draw
  - **1** = Loss to the home team
  - **2** = Win for the home team

## Models Experimented

### 1. **Random Forest:**
- **Description**: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification.
- **Implementation**: Used the `RandomForestClassifier` from the `sklearn.ensemble` module.
- **Hyperparameters**: Experimented with different numbers of trees (`n_estimators`), maximum depth of trees (`max_depth`), and other parameters to optimize the model's performance.

### 2. **Artificial Neural Network (ANN):**
- **Description**: ANN is a computational model inspired by the way biological neural networks in the human brain process information. It consists of layers of interconnected nodes (neurons).
- **Implementation**: Used the `Sequential` model from the `keras.models` module, with multiple dense layers.
- **Hyperparameters**: Experimented with different numbers of layers, neurons per layer, activation functions (e.g., ReLU, sigmoid), learning rates, and batch sizes to optimize the model's performance.

### 3. **Deep Learning Model:**
- **Description**: Deep Learning models are a subset of machine learning models with multiple layers that can learn representations of data with multiple levels of abstraction.
- **Implementation**: Used a deep neural network with multiple hidden layers, implemented using the `Sequential` model from the `keras.models` module.
- **Hyperparameters**: Experimented with different architectures, including varying the number of hidden layers, neurons per layer, activation functions, dropout rates, learning rates, and batch sizes to optimize the model's performance.

## Approach

### 1. **Data Preprocessing:**
- Loaded the datasets (`X_train.csv`, `y_train.csv`, `X_test.csv`) using pandas.
- Handled missing values and performed data cleaning.
- Conducted feature engineering and selection to improve model performance.

### 2. **Model Training:**
- Split the training data into training and validation sets.
- Trained the models with different hyperparameters and architectures.
- Used cross-validation to ensure the model's robustness and avoid overfitting.

### 3. **Model Evaluation:**
- Evaluated the models using metrics such as accuracy, precision, recall, and F1-score.
- Selected the best-performing model based on validation performance.

### 4. **Prediction:**
- Used the trained model to predict the target variable for the test dataset (`X_test.csv`).
- Saved the predicted values in the required format (`sample_submission.csv`).

This structured approach ensures a thorough analysis and robust solution to the classification problem.

## Competition Details:
- **File Size**: 3.4 MB
- **File Type**: CSV
- **License**: Subject to competition rules.

## How to Use:

1. **Download Data**: Download the dataset from the competition platform or Kaggle.
2. **Model Building**: Use the training set (`X_train.csv` and `y_train.csv`) to train a machine learning model.
3. **Model Evaluation**: Use the test set (`X_test.csv`) to predict the match results.
4. **Submit Predictions**: Format the predictions according to `sample_submission.csv` and submit them for evaluation.

## Competition Rules:
- You must agree to the competition rules to access the data.
- **Sign In** or **Register** on the competition platform to participate.

## Requirements:
- Python 3.x
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn` (or any other machine learning library you use)
  - `matplotlib` (for visualizations)

## License:
This project is subject to the competition rules and may not be used for any commercial purposes without permission.

