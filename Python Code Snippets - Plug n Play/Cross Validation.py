import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
import lightgbm as lgb
import xgboost as xgb
from rgf.sklearn import RGFRegressor, FastRGFRegressor
from sklearn.linear_model import Ridge, Lasso

# Performs cross validation using one of the XGBoost, LightGBM, RGFRegressor, 
# FastRGFRegressor, Ridge Regression, or Lasso Regression algorithm as specified
# Provides three methods to create train/validation splits
# 		Method 1: Time based splits - Create splits manually based on a date column
# 		Method 2: K-fold cross validation with data shuffled only once at the start. 
#                 Each test sets do not overlap. Most commonly used method
# 		Method 3: K-fold cross validation with data shuffled before each split. 
#                 Test sets may overlap between the splits

# -----------------------------------------------------------------------------
# 						   STEP 0 - SET VARIABLES
# -----------------------------------------------------------------------------
# File location for train data set
train_file_path = '../Data/train.csv'
# Target column to predict
target_column = 'visitors'
# Unnecessarry columns to be removed from train if any
rem_cols = ['air_store_id', 'air_area_name_mod']
# Method to be used to split data into folds.  
# Specify one of 'Manual', 'KFold', 'StratifiedKFold' or 'StratifiedShuffledKFold'
# 'StratifiedKFold' and 'StratifiedShuffledKFold' works for classification only
# For 'Regression', either choose 'Manual' or 'KFold'
# In case of 'Manual', split criteria need to be set manually afterwards
split_method = 'StratifiedShuffledKFold' 
# Column on which manual splitting is to be done if split_method is 'Manual'
date_col = 'visit_date'
# Specify the number of folds to be created
nfolds = 5 
# Specify if custom evaluation function is used for CV score instead of 
# built-in functions provided by some algorithms. '1' if used, otherwise '0'
custom_eval_used = 1
# Machine learning algorithm to be used. 
# Set one of 'XGBoost', 'LightGBM', 'RGF', 'FastRGF', 'Ridge', 'Lasso'
# Set the corresponding parameters for the chosen algorithm below
ml_algorithm = 'RGF' 
# Set machine learning algorithms parameters
# XGBoost
if ml_algorithm == 'XGBoost': 
	num_rounds = 10
	params = {'eta': 0.01, 'max_depth': 18, 'colsample_bytree': 0.2, 'subsample': 0.8, 
               'colsample_bylevel':0.3, 'alpha':2, 'seed': 99, 'silent': True, 
               'objective': 'reg:linear', 'eval_metric': 'rmse'}
                # 'objective': 'binary:logistic', 'eval_metric': 'auc'
# LightGBM
# Remove 'feval' parameter from LightGBM train method if custom evaluation function is not used
elif ml_algorithm == 'LightGBM': 
	num_rounds = 10
	params = {'learning_rate': 0.01, 'max_depth': 13, 'colsample_bytree': 0.2, 
               'num_leaves' : 580, 'objective': 'binary', 'metric': 'auc', 'seed': 99, 
               'silent': True}
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

# Define custom evaluation function to calucalte cross-validation score 
# Some models like LightGBM and XGBoost has built-in functions for evaluating 
# model performance as well
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

def gini_score(preds, y):
    score = eval_gini(y, preds) / eval_gini(y, y)
    return score

# For LightGBM
def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = eval_gini(y, preds) / eval_gini(y, y)
    return 'custom_eval_score', score, True
# -----------------------------------------------------------------------------
# 						STEP 1 - READ TRAIN DATA SET 
# -----------------------------------------------------------------------------
# Read train dataset
train = pd.read_csv(train_file_path)

# Remove redundant columns
train = train.drop(rem_cols, axis=1)

# -----------------------------------------------------------------------------
# 				  STEP 2 - CREATE TRAIN/VALIDATION SPLITS
# -----------------------------------------------------------------------------
data = {}
# Method 1: Time based splits - Create splits manually based on a date column
# data[0] - train; data[1] - validation; data[3] - train; data[4] - validation ...
if split_method == 'Manual':
	data = {
		0: train[(train[date_col] >= '2016-04-01') & (train[date_col] <= '2017-03-22')], # Fold 1: Train 
		1: train[(train[date_col] >= '2017-03-30') & (train[date_col] <= '2017-04-22')], # Fold 1: Validation
		2: train[(train[date_col] >= '2016-04-01') & (train[date_col] <= '2017-02-22')], # Fold 2: Train
		3: train[(train[date_col] >= '2017-02-28') & (train[date_col] <= '2017-03-31')], # Fold 2: Validation
		4: train[(train[date_col] >= '2016-04-01') & (train[date_col] <= '2017-01-22')], # Fold 3: Train
		5: train[(train[date_col] >= '2017-01-30') & (train[date_col] <= '2017-02-28')], # Fold 3: Validation 
		6: train[(train[date_col] >= '2016-04-01') & (train[date_col] <= '2017-03-14')], # Fold 4: Train
		7: train[(train[date_col] >= '2017-03-15') & (train[date_col] <= '2017-04-22')]  # Fold 4: Validation
		}
	for i in range(0, int(len(data.keys()))):
		data[j] = data[j].drop(date_col, axis=1)

# Method 2: K-fold cross validation. Does not preserves class balance
#           Can be used for both classification and regression
elif split_method == 'KFold':
    kfold_split = KFold(n_splits=nfolds, random_state=1, shuffle=True)

# Method 3: K-fold cross validation with data shuffled only once at the start
# 			Each test sets do not overlap. Preserves class balance
#           Can be used for classification only
elif split_method == 'StratifiedKFold':
	kfold_split = StratifiedKFold(n_splits=nfolds, random_state=1)

# Method 4: K-fold cross validation with data shuffled before each split
# 			Test sets may overlap between the splits. Preserves class balance
#           Can be used for classification only
elif split_method == 'StratifiedShuffledKFold':
	kfold_split = StratifiedShuffleSplit(n_splits=nfolds, test_size=0.5, random_state=1)

if split_method == 'KFold':
    i=0
    for train_index, test_index in kfold_split.split(train): 
       data[i] = train.iloc[train_index]
       data[i+1] = train.iloc[test_index]
       i=i+2   
elif split_method in ('StratifiedKFold', 'StratifiedShuffledKFold'):
    train_label = train[target_column]
    train = train.drop([target_column], axis=1)
    i=0
    for train_index, test_index in kfold_split.split(train, train_label):
        data[i] = pd.concat([train.iloc[train_index], train_label.iloc[train_index]], axis=1)
        data[i+1] = pd.concat([train.iloc[test_index], train_label.iloc[test_index]], axis=1)
        i=i+2     

# -----------------------------------------------------------------------------
# 					STEP 3 - PERFORM CROSS VALIDATION
# -----------------------------------------------------------------------------
cv_results = pd.DataFrame(columns=['fold', 'best_iteration', 'best_score'])
j = 0
for i in range(0, int(len(data.keys()) / 2)):
    x_train = pd.DataFrame(data[j])
    x_valid = pd.DataFrame(data[j + 1])

	# Separate target variable
    y_train = x_train[target_column].values
    y_valid = x_valid[target_column].values  
    x_train = x_train.drop(target_column, axis=1)
    x_valid = x_valid.drop(target_column, axis=1)   
   
    print('Training ...', i+1)
	# XGBoost
    if ml_algorithm == 'XGBoost': 
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(params, d_train, num_rounds, watchlist, 
                          early_stopping_rounds=100, verbose_eval=10)		
        # Custom scoring function (optional)
        y_pred = model.predict(d_valid, model.best_iteration)
        custom_eval_score = gini_score(y_pred, y_valid)
	# LightGBM
    elif ml_algorithm == 'LightGBM': 
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid, reference=d_train)
        model = lgb.train(params, d_train, num_rounds, valid_sets=d_valid, 
                          feval=gini_lgb, early_stopping_rounds=100, verbose_eval=10)		
	# RGFRegressor, FastRGFRegressor, Ridge Regression, Lasso Regression
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid) 
        custom_eval_score = gini_score(y_pred, y_valid)

    if custom_eval_used == 1 and ml_algorithm == 'LightGBM':
        cv_results = cv_results.append(pd.DataFrame(
                [[i+1, model.best_iteration, 
                  model.best_score['valid_0']['custom_eval_score']]], 
                columns = ['fold', 'best_iteration', 'best_score']))         
    elif custom_eval_used == 1 and ml_algorithm == 'XGBoost': 
        cv_results = cv_results.append(pd.DataFrame(
                [[i+1, model.best_iteration, custom_eval_score]], 
                columns = ['fold', 'best_iteration', 'best_score'])) 
    elif ml_algorithm == 'LightGBM':
        cv_results = cv_results.append(pd.DataFrame(
                [[i+1, model.best_iteration, 
                  model.best_score[next(iter(model.best_score))]
                  [next(iter(model.best_score[next(iter(model.best_score))]))]]], 
                columns = ['fold', 'best_iteration', 'best_score']))          
    elif ml_algorithm == 'XGBoost':     
        cv_results = cv_results.append(pd.DataFrame(
                [[i+1, model.best_iteration, model.best_score]], 
                columns = ['fold', 'best_iteration', 'best_score'])) 
    else:
        cv_results = cv_results.append(pd.DataFrame(
                [[i+1, 'N/A', custom_eval_score]], 
                columns = ['fold', 'best_iteration', 'best_score']))         
    j = j + 2

# -----------------------------------------------------------------------------
# 					STEP 4 - AVERAGE ALL FOLDS SCORE
# -----------------------------------------------------------------------------
cv_results = cv_results.append(pd.DataFrame(
        [['Avg First ' + str(int(len(data.keys())/2)-1) + ' Folds', '', 
          np.mean(cv_results.best_score[0:(int(len(data.keys())/2)-1)])]], 
          columns = ['fold', 'best_iteration', 'best_score']))
cv_results = cv_results.append(pd.DataFrame(
        [['Avg All Folds', '', 
          np.mean(cv_results.best_score[0:(int(len(data.keys())/2))])]], 
          columns = ['fold', 'best_iteration', 'best_score']))