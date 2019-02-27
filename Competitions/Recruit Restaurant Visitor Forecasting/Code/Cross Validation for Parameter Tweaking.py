import numpy as np
import pandas as pd
import xgboost as xgb

# -------------------------------------------------------------------------------------------------------------------------------------
# 										READ TRAIN DATA SET AND CREATE TRAIN/VALIDATION SPLITS
# -------------------------------------------------------------------------------------------------------------------------------------
# Read train dataset
train = pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\train.csv')

# Remove redundant columns
train = train.drop(['air_store_id', 'air_area_name_mod', 'air_genre_name_mod'], axis=1)

# Create Train / Validation splits
data = {
    0: train[(train.visit_date >= '2016-04-01') & (train.visit_date <= '2017-03-22')], # Fold 1: Train 
	1: train[(train.visit_date >= '2017-03-30') & (train.visit_date <= '2017-04-22')], # Fold 1: Validation
	2: train[(train.visit_date >= '2016-04-01') & (train.visit_date <= '2017-02-22')], # Fold 2: Train
	3: train[(train.visit_date >= '2017-02-28') & (train.visit_date <= '2017-03-31')], # Fold 2: Validation
	4: train[(train.visit_date >= '2016-04-01') & (train.visit_date <= '2017-01-22')], # Fold 3: Train
	5: train[(train.visit_date >= '2017-01-30') & (train.visit_date <= '2017-02-28')], # Fold 3: Validation 
	6: train[(train.visit_date >= '2016-04-01') & (train.visit_date <= '2017-03-14')], # Fold 4: Train
	7: train[(train.visit_date >= '2017-03-15') & (train.visit_date <= '2017-04-22')]  # Fold 4: Validation
    }

# -------------------------------------------------------------------------------------------------------------------------------------
# 										CROSS VALIDATION FOR PARAMETER TWEAKING
# -------------------------------------------------------------------------------------------------------------------------------------
def xgb_param_tweaking_cv(param_to_be_tweaked, param_value): 
    # Set XGBoost parameters
    params = {'eta': 0.01, 'max_depth': 18, 'colsample_bytree': 0.2, 'subsample': 0.8, 'colsample_bylevel':0.3, 'alpha':2,
                'objective': 'reg:linear', 'eval_metric': 'rmse', 'seed': 99, 'silent': True}
    # Set parameter to be tweaked
    params[param_to_be_tweaked] = param_value

    cv_results = pd.DataFrame(columns=['param_name', 'param_value', 'type', 'best_iteration', 'best_score'])
    j = 0
    for i in range(0, int(len(data.keys()) / 2)):       
        x_train = pd.DataFrame(data[j])
        x_valid = pd.DataFrame(data[j + 1])
        
    	# Separate target variable
        y_train = x_train.visitors.values
        y_valid = x_valid.visitors.values  
        x_train = x_train.drop(['visitors', 'visit_date'], axis=1)
        x_valid = x_valid.drop(['visitors', 'visit_date'], axis=1)  
        
        print('Building DMatrix...')
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)
        
        print('Training ', i+1, ' ... ', param_to_be_tweaked, '...', param_value)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        gbdt = xgb.train(params, d_train, 1000000, watchlist, early_stopping_rounds=100, verbose_eval=10)
        
        cv_results = cv_results.append(pd.DataFrame([[param_to_be_tweaked, param_value, str(i+1) + ' Fold', gbdt.best_iteration, gbdt.best_score]], columns = ['param_name', 'param_value', 'type', 'best_iteration', 'best_score']))
        j = j + 2
   
    # Average all folds score
    cv_results = cv_results.append(pd.DataFrame([[param_to_be_tweaked, param_value, 'Avg First ' + str(int(len(data.keys())/2)-1) + ' Folds', '', np.mean(cv_results.best_score[0:(int(len(data.keys())/2)-1)])]], columns = ['param_name', 'param_value', 'type', 'best_iteration', 'best_score']))
    cv_results = cv_results.append(pd.DataFrame([[param_to_be_tweaked, param_value, 'Avg All Folds', '', np.mean(cv_results.best_score[0:int(len(data.keys())/2)])]], columns = ['param_name', 'param_value', 'type', 'best_iteration', 'best_score']))

    return cv_results

# -------------------------------------------------------------------------------------------------------------------------------------
# 									CALL FUNCTION FOR DIFFERENT PARAMETER VALUES
# -------------------------------------------------------------------------------------------------------------------------------------
cv_results_out = xgb_param_tweaking_cv('eta', 0.03)
cv_results_out = cv_results_out.append(xgb_param_tweaking_cv('eta', 0.05))
cv_results_out = cv_results_out.append(xgb_param_tweaking_cv('eta', 0.06))

