import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import tqdm
import time

# load the data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# generate 1500 random data of 10 hyperparameters
print("Generating random hyperparameters for testing data...")
np.random.seed(1)
n_samples = 750
params = {
    'learning_rate': np.random.uniform(0.01, 0.3, n_samples),
    'n_estimators': np.random.randint(100, 1000, n_samples).astype(int),
    'max_depth': np.random.randint(3, 10, n_samples).astype(int),
    'min_child_weight': np.random.randint(1, 6, n_samples).astype(int),
    'subsample': np.random.uniform(0.6, 1, n_samples),
    'gamma': np.random.uniform(0, 0.5, n_samples),
    'alpha': np.random.uniform(0, 1, n_samples),
    'lambda': np.random.uniform(0, 1, n_samples),
    'colsample_bytree': np.random.uniform(0.6, 1, n_samples),
    'scale_pos_weight': np.random.randint(1, 5, n_samples).astype(int),
}

assert params['n_estimators'].dtype == int, "n_estimators should be an integer"
assert params['max_depth'].dtype == int, "max_depth should be an integer"
assert params['min_child_weight'].dtype == int, "min_child_weight should be an integer"
assert params['scale_pos_weight'].dtype == int, "scale_pos_weight should be an integer"


# save a .csv file with the hyperparameters
params_df = pd.DataFrame(params)

# train function
x_train, y_train = train.drop('Depression', axis=1), train['Depression']
x_test, y_test = test.drop('Depression', axis=1), test['Depression']

def train_and_evaluate(params: dict, index: int):
    '''
    train and evaluate a XGBoost model with the given hyperparameters

    Args:
        params (dict): a set of hyperparameters for XGBoost
        index (int): the index of the hyperparameters in the params_df
    '''
    model = XGBClassifier(
        learning_rate = params['learning_rate'],
        n_estimators = params['n_estimators'],
        max_depth = params['max_depth'],
        min_child_weight = params['min_child_weight'],
        subsample = params['subsample'],
        gamma = params['gamma'],
        reg_alpha = params['alpha'],
        reg_lambda = params['lambda'],
        colsample_bytree = params['colsample_bytree'],
        scale_pos_weight = params['scale_pos_weight'],
        objective = 'binary:logistic',
        random_state = 42,
    )
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)

    # Save the hyperparameters and f1 score to params_df
    params_df.loc[index, 'F1'] = f1

print("Training and evaluating the models for testing data...")

# calculate elapse time
start_time = time.time()

# train and evaluate the real data iteratively
for index, row in tqdm.tqdm(params_df.iterrows(), total=n_samples, desc="Training"):
    param_dict = row.to_dict()
    param_dict['n_estimators'] = int(param_dict['n_estimators'])
    param_dict['max_depth'] = int(param_dict['max_depth'])
    param_dict['min_child_weight'] = int(param_dict['min_child_weight'])
    param_dict['scale_pos_weight'] = int(param_dict['scale_pos_weight'])
    train_and_evaluate(param_dict, index)

# calculate elapse time
end_time = time.time()
print("Training and evaluating the models took {:.2f} seconds".format(end_time - start_time))

# save the params_df to a .csv file
print("Saving the results...")
params_df.to_csv('data/results.csv', index=False)

print("Done!")