import numpy as np
import pandas as pd
import xgboost as xgb
import operator

# -----------------------------------------------------------------------------
# 						   		STEP 0 - SET VARIABLES
# -----------------------------------------------------------------------------
# File location for train data set
train_file_path = '../Data/train.csv'
# Target column to predict
target_column = 'visitors'
# File location to save feature importance from XGBoost
xgboost_feat_imp_file_path = '../Data/feat_imp.csv'
# Remove redundant columns from train (if any). Should not include target column
rem_cols = ['air_store_id', 'visit_date', 'air_genre_name_mod', 'air_area_name_mod']
# Set thresold value for inter predictor correlation
inter_predictor_corr_thresh = 0.95
# Set thresold value for predictor correlation with the target
inter_corr_with_tgt_feat_thresh = 1
# Set parameters for XGBoost 
num_rounds = 10
params = {'eta': 0.01, 'max_depth': 18, 'colsample_bytree': 0.2, 'subsample': 0.8, 
		   'colsample_bylevel':0.3, 'alpha':2, 'objective': 'reg:linear', 
		   'eval_metric': 'rmse', 'seed': 99, 'silent': True}
			# 'objective': 'binary:logistic', 'eval_metric': 'auc'
			
# -----------------------------------------------------------------------------
# 			STEP 1 - READ TRAIN DATA SET AND REMOVE REDUNDANT COLUMNS
# -----------------------------------------------------------------------------
train = pd.read_csv(train_file_path)
train = train.drop(rem_cols, axis=1)

# -----------------------------------------------------------------------------
# 		STEP 2 - GET FEATURES WITH INTER-PREDICTOR CORRELATION >= THRESHOLD
# -----------------------------------------------------------------------------
def get_high_inter_predictor_corr_feat(data, target_col, threshold):   
    # Compute correlation matrix
    corr_matrix = data.corr().abs()     
    
    # Find variables with inter-predictor correlation >= threshold
    high_corr_var = np.where(corr_matrix >= threshold)
    high_corr_var = pd.DataFrame([(corr_matrix.index[x], corr_matrix.columns[y]) 
									for x, y in zip(*high_corr_var) if x != y and x < y])
  
    # Get features with inter-predictor correlation >= threshold and less correlation with target
    for i in range(high_corr_var.count()[0]):
        high_corr_var.loc[i,2] = corr_matrix.loc[high_corr_var.iloc[i,0], target_col]
        high_corr_var.loc[i,3] = corr_matrix.loc[high_corr_var.iloc[i,1], target_col]
        
        if (high_corr_var.loc[i,2] < high_corr_var.loc[i,3]):
            high_corr_var.loc[i,4] = high_corr_var.iloc[i,0]
        else:
            high_corr_var.loc[i,4] = high_corr_var.iloc[i,1]
     
    final_cols = pd.DataFrame(high_corr_var.loc[:,4].unique())
    return final_cols

high_inter_predictor_corr_feat = get_high_inter_predictor_corr_feat(train, \
									target_column, inter_predictor_corr_thresh)

# ------------------------------------------------------------------------------
# 		STEP 3 - GET FEATURES HAVING LEAST CORRELATION WITH TARGET
# ------------------------------------------------------------------------------
def rem_least_corr_with_target(data, target_col, num_feat):   
    # Compute correlation matrix
    corr_matrix = data.corr().abs() 
    # Sort correlation matrix in ascending order w.r.t correlation with target
    corr_matrix = corr_matrix.sort_values(by=target_col, ascending=True) 
    # Get features with least correlation with target
    final_cols = pd.DataFrame(corr_matrix.iloc[:num_feat].index.values) 
    return final_cols

least_corr_with_target_feat = rem_least_corr_with_target(train, target_column, \
												inter_corr_with_tgt_feat_thresh)

# -------------------------------------------------------------------------------
# 			STEP 4 - GET FEATURE IMPORTANCE FROM XGBOOST MODEL
# -------------------------------------------------------------------------------
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

y_train = train[target_column].values
x_train = train.drop([target_column], axis=1)

print('Building DMatrix...')
d_train = xgb.DMatrix(x_train, label=y_train)

print('Training XGBoost Model...')
model = xgb.train(params, d_train, num_rounds, verbose_eval=10)
    
# Get fscore of all features from XGBoost trained model  
feat = x_train.columns
ceate_feature_map(feat)
importance = model.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

# Plot features importance
xgb.plot_importance(model)

# Save features importance to csv file
df.to_csv(xgboost_feat_imp_file_path, index=False)