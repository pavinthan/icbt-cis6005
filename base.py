import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew
from lib import create_submission_file, get_outer_ids, RMSE

def preprocess_train_test_data(train,test):
    outlier_ids = get_outer_ids()
    train.drop(train.index[outlier_ids],inplace=True)
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))
    
    to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
    all_data = all_data.drop(to_delete,axis=1)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    #log transform skewed numeric features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    x_train = all_data[:train.shape[0]]
    x_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return x_train,x_test,y
    
def model_random_forecast(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain

    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    model = GridSearchCV(estimator=rfr, param_grid={}, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)

    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
    return model

def model_gradient_boosting_tree(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 

    gbr = GradientBoostingRegressor(random_state=0)
    model = GridSearchCV(estimator=gbr, param_grid={}, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
    return model

def model_xgb_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    
    xgbreg = xgb.XGBRegressor(seed=0)
    model = GridSearchCV(estimator=xgbreg, param_grid={}, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    
    print('eXtreme Gradient Boosting regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
    return model

def model_extra_trees_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    model = GridSearchCV(estimator=etr, param_grid={}, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    
    print('Extra trees regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)
    
    return model

# Main function
def main():
    # read train data
    train = pd.read_csv("./data/train.csv")

    # read test data
    test = pd.read_csv("./data/test.csv")
    
    X_train, X_test, y_train = preprocess_train_test_data(train,test)
    
    models = [model_random_forecast, model_xgb_regression, model_extra_trees_regression, model_gradient_boosting_tree]
    model_names = ["random-forecast", "xgb-regression", "extra-trees-regression", "gradient-boosting-tree"]

    for create_model, model_name in zip(models, model_names):
        model = create_model(X_train, X_test, y_train)

        y_pred = model.predict(X_test)
        
        create_submission_file(model_name, test, np.exp(y_pred), -model.best_score_)
        
        # Save to file in the current working directory
        pkl_filename = f'output/{model_name}.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

# Call the main function
if __name__ == "__main__":
    main()
