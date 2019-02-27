import numpy as np
import pandas as pd
import operator

# -------------------------------------------------------------------------------------------------------------------------------------
# 												READ TRAIN DATA SET
# -------------------------------------------------------------------------------------------------------------------------------------
train = pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\train.csv')

# -------------------------------------------------------------------------------------------------------------------------------------
# 							1. GET FEATURES WITH INTER-PREDICTOR CORRELATION >= THRESHOLD
# -------------------------------------------------------------------------------------------------------------------------------------
def get_high_inter_predictor_corr_feat(data, target_col, threshold):   
    # Compute correlation matrix
    corr_matrix = data.corr().abs()     
    
    # Find variables with inter-predictor correlation >= threshold
    high_corr_var = np.where(corr_matrix >= threshold)
    high_corr_var = pd.DataFrame([(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_var) if x != y and x < y])
  
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

high_inter_predictor_corr_feat = get_high_inter_predictor_corr_feat(train, 'visitors', 0.95)

# -------------------------------------------------------------------------------------------------------------------------------------
# 							2. GET FEATURES HAVING LEAST CORRELATION WITH TARGET
# -------------------------------------------------------------------------------------------------------------------------------------
def rem_least_corr_with_target(data, target_col, num_feat):   
    # Compute correlation matrix
    corr_matrix = data.corr().abs() 
    # Sort correlation matrix in ascending order w.r.t correlation with target
    corr_matrix = corr_matrix.sort_values(by=target_col, ascending=True) 
    # Get features with least correlation with target
    final_cols = pd.DataFrame(corr_matrix.iloc[:num_feat].index.values) 
    return final_cols

least_corr_with_target_feat = rem_least_corr_with_target(train, 'visitors', 1)

# -------------------------------------------------------------------------------------------------------------------------------------
# 							3. GET FEATURE IMPORTANCE FROM XGBOOST MODEL
# -------------------------------------------------------------------------------------------------------------------------------------
def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

# Get list of feature names
feat = x_train.columns
    
# Get fscore of all features from XGBoost trained model  
ceate_feature_map(feat)
importance = gbdt.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

# Plot features importance
xgb.plot_importance(gbdt)

# Save features importance to csv file
df.to_csv('C:\\Users\\muhamadatif.sajjad\\Dropbox\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\FeatureSelection\\FeatImportanceSecondDataset.csv', index=False)
