# Hyperparameter Optimization for XGBoost Classification

This project performs hyperparameter optimization for an XGBoost binary classification model. It generates random hyperparameters, trains the model on a dataset, and evaluates the model's performance using F1-score. The goal is to analyze the relationship between hyperparameters and F1-score using a regression model.

## Project Overview

The project generates a set of 1500 random hyperparameter combinations for an XGBoost classifier. The model is then trained on a dataset with features, and the F1-score is computed on the test set. The results, including the hyperparameters and their corresponding F1-scores, are saved to **results.csv** for further analysis. 

Then, we conduct a statistical analysis to determine the relationship between hyperparameters and F1-score. We use a stepwise regression model to fit a linear regression model to the data. The objective is to determine which hyperparameters are most important in predicting F1-score.

## Data

- **train.csv**: The training dataset, including features and target variable (`Depression`).
- **test.csv**: The test dataset used to evaluate the model's performance.
- **results.csv**: The output file containing the hyperparameters and their corresponding F1-scores.

## Hyperparameters

The following hyperparameters are randomly generated and optimized:
- `learning_rate`: The learning rate for the gradient boosting model.
- `n_estimators`: The number of boosting rounds (trees).
- `max_depth`: The maximum depth of each tree.
- `min_child_weight`: The minimum sum of instance weight (hessian) in a child.
- `subsample`: The fraction of samples used for each boosting round.
- `gamma`: The minimum loss reduction required to make a further partition.
- `alpha`: L1 regularization term.
- `lambda`: L2 regularization term.
- `colsample_bytree`: The fraction of features used for each tree.
- `scale_pos_weight`: The balance of positive and negative weights for imbalanced classes.

## Requirements

- Python 3.x
- `xgboost`
- `pandas`
- `numpy`
- `sklearn`
- `tqdm`

