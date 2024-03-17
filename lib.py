from sklearn.metrics import make_scorer, mean_squared_error
import pandas as pd
import json

def create_submission_file(model, test, prediction,score):
    submission_file_name = f'output/{model}-{score}.csv'
    # submission_file_name = 'prediction.csv'
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(submission_file_name, index=False)

def calculate_root_mean_squared_error(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

def get_outer_ids():
    try:
        with open('outlier_ids.json') as f:
            data = json.load(f)
            return data
    except:
        return []

RMSE = make_scorer(calculate_root_mean_squared_error, greater_is_better=False)    
