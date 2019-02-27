import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from rgf.sklearn import RGFRegressor, FastRGFRegressor
from sklearn.linear_model import Ridge, Lasso

# Use one algorithm (XGBoost, LightGBM, RGFRegressor, FastRGFRegressor, 
# Ridge Regression or Lasso Regression) at a time to train and predict
# Makes predictions for one target column
# Kaggle submission file will be generated with two columns - ID and predictions

# -----------------------------------------------------------------------------
# 							STEP 0 - SET VARIABLES
# -----------------------------------------------------------------------------
# File location for train data set
train_file_path = '../Data/train.csv'
# File location for test data set
test_file_path = '../Data/test.csv'
# File location for generating the kaggle submission file
submission_file_path = '../Submissions/XGB_Prediction.csv'
# Specify ID column which uniquely identifies a row
id_column = 'air_store_id'
# Specify target column name for which prediction is to be made
target_column = 'visitors'
# ID column name in submission file
id_column_label = 'id'
# Target column name in submission file
target_column_label = 'visitors'
# Unnecessarry columns to be removed from datasets if any
rem_cols = ['mean_visitors_lat_plus_long_dow', 'median_visitors_area2_dow']
# Set one of 'XGBoost', 'LightGBM', 'RGF', 'FastRGF', 'Ridge', 'Lasso'
# Set the corresponding parameters for the chosen algorithm below
ml_algorithm = 'XGBoost' 
# Set machine learning algorithms parameters
# XGBoost
if ml_algorithm == 'XGBoost': 
	num_rounds = 10
	params = {'eta': 0.01, 'max_depth': 18, 'colsample_bytree': 0.2, 'subsample': 0.8, 
               'colsample_bylevel':0.3, 'alpha':2, 'objective': 'reg:linear', 
               'eval_metric': 'rmse', 'seed': 99, 'silent': True}
                # 'objective': 'binary:logistic', 'eval_metric': 'auc'
# LightGBM
elif ml_algorithm == 'LightGBM': 
	num_rounds = 10
	params = {'learning_rate': 0.01, 'max_depth': 13, 'colsample_bytree': 0.2, 
               'num_leaves' : 580, 'application': 'regression', 'metric': 'rmse', 
               'seed': 99, 'silent': True}
# RGFRegressor 
elif ml_algorithm == 'RGF':
	model = RGFRegressor(max_leaf=3500, algorithm='RGF_Opt', loss="LS", l2=0.01)  
# FastRGFRegressor 
elif ml_algorithm == 'FastRGF':     
	model = FastRGFRegressor(n_estimators=1200, sparse_max_features=1500, max_depth=5, 
                              max_bin=150, min_samples_leaf=12, sparse_min_occurences=1, 
                              opt_algorithm='epsilon-greedy', l2=1.0, min_child_weight=210.0, 
								learning_rate=0.2) 
# Ridge Regression
elif ml_algorithm == 'Ridge': 
	model = Ridge(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100, 
                   normalize=False, random_state=101, solver='auto', tol=0.01)
# Lasso Regression
elif ml_algorithm == 'Lasso': 
	model = Lasso(alpha=.6, copy_X=True, fit_intercept=True, max_iter=100, 
                   normalize=False, random_state=101, tol=0.01)
	
# -----------------------------------------------------------------------------
# 							STEP 1 - READ DATA FILES
# -----------------------------------------------------------------------------
train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

# -----------------------------------------------------------------------------
# 					STEP 2 - REMOVE REDUNDANT COLUMNS IF ANY
# -----------------------------------------------------------------------------
train = train.drop(rem_cols, axis=1)
test = test.drop(rem_cols, axis=1)
    
# -----------------------------------------------------------------------------
# 			STEP 3 - PREPARE TEST, TRAIN, AND KAGGLE SUBMISSION
# -----------------------------------------------------------------------------
# Kaggle submission file
test_id = test[id_column]
submission = pd.DataFrame(columns=[id_column_label, target_column_label])

# Separate target variable and features in train data set
y_train = train[target_column].values
x_train = train.drop([target_column], axis=1)

# Drop irrelevant columns from train and test
x_train = x_train.drop([id_column], axis=1)
test = test.drop([id_column], axis=1)   

# -----------------------------------------------------------------------------
# 			STEP 4 - TRAIN ML MODEL AND GENERATE PREDICTIONS
# -----------------------------------------------------------------------------
# XGBoost
if ml_algorithm == 'XGBoost': 
	d_train = xgb.DMatrix(x_train, label=y_train)
	d_test = xgb.DMatrix(test)
	model = xgb.train(params, d_train, num_rounds, verbose_eval=10)
	prediction = model.predict(d_test)
# LightGBM
elif ml_algorithm == 'LightGBM': 
	d_train = lgb.Dataset(x_train, label=y_train)
	d_test = lgb.Dataset(test)
	model = lgb.train(params, d_train, num_rounds, verbose_eval=10)
	prediction = model.predict(d_test)	
# RGFRegressor, FastRGFRegressor, Ridge Regression, Lasso Regression
else:
	model.fit(x_train, y_train)
	prediction = model.predict(test) 

# -----------------------------------------------------------------------------
# 				STEP 5 - GENERATE KAGGLE SUBMISSION FILE
# -----------------------------------------------------------------------------
print('Generate Submission ...')
submission = submission.append(pd.DataFrame(
                {id_column_label: test_id, target_column_label: prediction}))
submission.to_csv(submission_file_path, index=False)