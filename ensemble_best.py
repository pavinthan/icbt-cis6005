import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import skew
from lib import create_submission_file, get_outer_ids, RMSE

# Preprocessing function
def preprocess_train_test_data(train,test):
    outlier_ids = get_outer_ids()
    train.drop(train.index[outlier_ids],inplace=True)
    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']))
    
    to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
    all_data = all_data.drop(to_delete,axis=1)

    train["SalePrice"] = np.log1p(train["SalePrice"])
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) 
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.SalePrice

    return X_train,X_test,y

# Ensemble Model class
class EnsembleModel(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, train, test, ytr):
        X = train.values
        y = ytr.values
        T = test.values
        folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=0).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                model.fit(X_train, y_train)
                S_train[test_idx, i] = model.predict(X_holdout)
                S_test_i[:, j] = model.predict(T)
            S_test[:, i] = S_test_i.mean(1)

        param_grid = {'alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.8, 1e0, 3, 5, 7, 1e1, 2e1, 5e1]}
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, cv=5, scoring=RMSE)
        grid.fit(S_train, y)

        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
        except:
            pass

        y_pred = grid.predict(S_test)
        return y_pred, -grid.best_score_

# Main function
def main():
    # read train data
    train = pd.read_csv("./data/train.csv")

    # read test data
    test = pd.read_csv("./data/test.csv")

    base_models = [
        RandomForestRegressor(n_jobs=1, random_state=0, n_estimators=500, max_features=18, max_depth=11),
        ExtraTreesRegressor(n_jobs=1, random_state=0, n_estimators=500, max_features=20),
        GradientBoostingRegressor(random_state=0, n_estimators=500, max_features=10, max_depth=6, learning_rate=0.05, subsample=0.8),
        XGBRegressor(seed=0, n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.75),
    ]

    ensemble_model = EnsembleModel(n_folds=5, stacker=Ridge(), base_models=base_models)

    X_train,X_test,y_train = preprocess_train_test_data(train,test)
    y_pred, score = ensemble_model.fit_predict(X_train,X_test,y_train)

    create_submission_file("ensemble-best", test, np.expm1(y_pred),score)
    
    # Save to file in the current working directory
    pkl_filename = "output/ensemble_best.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(ensemble_model, file)

# Call the main function
if __name__ == "__main__":
    main()
