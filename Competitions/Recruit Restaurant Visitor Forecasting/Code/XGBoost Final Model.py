import pandas as pd
import numpy as np
import xgboost as xgb

# -------------------------------------------------------------------------------------------------------------------------------------
# 												STEP 0 - READ DATA FILES
# -------------------------------------------------------------------------------------------------------------------------------------
train = pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\train.csv')
test = pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\test.csv')

# -------------------------------------------------------------------------------------------------------------------------------------
# 								STEP 1 - REMOVE REDUNDANT COLUMNS IDENTIFIED THROUGH CROSS-VALIDATION
# -------------------------------------------------------------------------------------------------------------------------------------
rem_cols = ['mean_visitors_lat_plus_long_dow', 'median_visitors_area2_dow', 'median_visitors_lat_plus_long_dow', 'size_grp_area', 'size_grp_store', 'year', 'total_reserved_visitors_x','total_reserved_visitors_y','avg_reserved_visitors_x','avg_reserved_visitors_y','avg_reserve_datetime_diff_x','avg_reserve_datetime_diff_y','total_reserve_datetime_diff_x','total_reserve_datetime_diff_y','total_reserved_visitors','avg_reserved_visitors','avg_reserve_datetime_diff', 'min_visitors_area2_dow', 'mean_visitors_area2_dow', 'max_visitors_area2_dow', 'month_of_year', 'min_visitors_lat_plus_long_dow', 'max_visitors_lat_plus_long_dow', 'air_area_name', 'min_visitors_store_holiday_dow', 'mean_visitors_store_dow', 'mean_visitors_store_holiday_dow', 'median_visitors_store_holiday_dow', 'median_visitors_store_dow', 'max_visitors_store_dow', 'max_visitors_store_holiday_dow', 'mean_visitors_genre_dow', 'mean_visitors_genre_holiday_dow', 'median_visitors_genre_holiday_dow', 'median_visitors_genre_dow', 'max_visitors_genre_holiday_dow', 'min_visitors_area_holiday_dow', 'mean_visitors_area_dow', 'mean_visitors_area_holiday_dow', 'median_visitors_area_holiday_dow', 'median_visitors_area_dow', 'max_visitors_area_dow', 'max_visitors_area_holiday_dow', 'min_visitors_area_genre_holiday_dow', 'max_visitors_area_genre_holiday_dow', 'mean_visitors_area_genre_dow', 'mean_visitors_area_genre_holiday_dow', 'median_visitors_area_genre_dow', 'median_visitors_area_genre_holiday_dow', 'min_visitors_store_holiday_weekend_weekday', 'mean_visitors_store_weekend_weekday', 'mean_visitors_store_holiday_weekend_weekday', 'median_visitors_store_holiday_weekend_weekday', 'median_visitors_store_weekend_weekday', 'max_visitors_store_holiday_weekend_weekday', 'max_visitors_store_first_ten_days_of_month', 'max_visitors_store_weekend_weekday', 'mean_visitors_genre_weekend_weekday', 'mean_visitors_genre_holiday_weekend_weekday', 'median_visitors_genre_holiday_weekend_weekday', 'median_visitors_genre_weekend_weekday', 'mean_visitors_area_weekend_weekday', 'mean_visitors_area_holiday_weekend_weekday', 'median_visitors_area_holiday_weekend_weekday', 'max_visitors_area_holiday_weekend_weekday', 'max_visitors_area_holiday', 'max_visitors_area_first_ten_days_of_month', 'max_visitors_area_weekend_weekday', 'max_visitors_area_last_15_days_of_month', 'max_visitors_area_genre_holiday_weekend_weekday', 'max_visitors_area_genre_weekend_weekday', 'max_visitors_genre_area_mid_ten_days_of_month', 'mean_visitors_area_genre_weekend_weekday', 'mean_visitors_area_genre_holiday_weekend_weekday', 'median_visitors_area_genre_holiday_weekend_weekday', 'median_visitors_area_genre_weekend_weekday', 'mean_visitors_area_holiday', 'mean_visitors_area_first_ten_days_of_month', 'median_visitors_area_holiday', 'mean_visitors_area_mid_ten_days_of_month', 'mean_visitors_area_last_ten_days_of_month', 'mean_visitors_area_first_15_days_of_month', 'mean_visitors_area_last_15_days_of_month', 'max_visitors_genre_area_first_ten_days_of_month', 'max_visitors_genre_area_last_ten_days_of_month', 'max_visitors_genre_area_first_15_days_of_month', 'max_visitors_genre_area_last_15_days_of_month', 'mean_visitors_area_genre_holiday_flg', 'mean_visitors_genre_area_first_ten_days_of_month', 'median_visitors_area_genre_holiday_flg', 'mean_visitors_genre_area_mid_ten_days_of_month', 'mean_visitors_genre_area_last_ten_days_of_month', 'mean_visitors_genre_area_first_15_days_of_month', 'mean_visitors_genre_area_last_15_days_of_month', 'min_visitors_store_first_ten_days_of_month', 'mean_visitors_store_first_ten_days_of_month', 'mean_visitors_store_mid_ten_days_of_month', 'mean_visitors_store_first_15_days_of_month', 'mean_visitors_store_last_15_days_of_month', 'median_visitors_store_mid_ten_days_of_month', 'mean_visitors_store_last_ten_days_of_month', 'median_visitors_store_first_ten_days_of_month', 'median_visitors_store_first_15_days_of_month', 'median_visitors_store_last_15_days_of_month', 'max_visitors_store_mid_ten_days_of_month', 'max_visitors_store_first_15_days_of_month', 'min_visitors_store_last_ten_days_of_month', 'max_visitors_store_last_ten_days_of_month', 'min_visitors_genre_first_ten_days_of_month', 'mean_visitors_genre_first_ten_days_of_month', 'mean_visitors_genre_mid_ten_days_of_month', 'mean_visitors_genre_first_15_days_of_month', 'mean_visitors_genre_last_15_days_of_month', 'median_visitors_genre_first_ten_days_of_month', 'mean_visitors_genre_last_ten_days_of_month', 'max_visitors_genre_first_15_days_of_month', 'median_visitors_genre_mid_ten_days_of_month', 'median_visitors_genre_first_15_days_of_month', 'median_visitors_genre_last_15_days_of_month', 'max_visitors_genre_last_15_days_of_month', 'min_visitors_genre_first_15_days_of_month', 'median_visitors_area_mid_ten_days_of_month', 'median_visitors_area_last_ten_days_of_month', 'median_visitors_area_first_15_days_of_month', 'median_visitors_area_last_15_days_of_month', 'min_visitors_area_first_15_days_of_month', 'min_visitors_area_last_15_days_of_month', 'max_visitors_area_mid_ten_days_of_month', 'max_visitors_area_first_15_days_of_month', 'min_visitors_area_last_ten_days_of_month', 'median_visitors_genre_area_mid_ten_days_of_month', 'median_visitors_genre_area_first_ten_days_of_month', 'median_visitors_genre_area_first_15_days_of_month', 'min_visitors_genre_area_last_ten_days_of_month', 'median_visitors_genre_area_last_15_days_of_month', 'min_visitors_genre_mid_ten_days_of_month', 'max_visitors_genre_last_ten_days_of_month', 'max_visitors_genre_holiday_weekend_weekday', 'is_first_mid_last_ten_days', 'air_genre_name_mod', 'air_area_name_mod']
train = train.drop(rem_cols, axis=1)
test = test.drop(rem_cols, axis=1)
    
# -------------------------------------------------------------------------------------------------------------------------------------
# 										STEP 2 - PREPARE TEST, TRAIN, AND KAGGLE SUBMISSION
# -------------------------------------------------------------------------------------------------------------------------------------
# Kaggle submission file
predictions = pd.DataFrame(columns=['id', 'visitors'])
test['visit_date'] = test['visit_date'].astype(str)
test_pk = test[['air_store_id', 'visit_date']].apply('_'.join, axis=1) 

# Separate target variable and features in train data set
y_train = train.visitors.values
x_train = train.drop(['visitors'], axis=1)

# Drop irrelevant columns from train and test
x_train = x_train.drop(['air_store_id', 'visit_date'], axis=1)
test = test.drop(['air_store_id', 'visit_date'], axis=1)   

# -------------------------------------------------------------------------------------------------------------------------------------
# 										STEP 3 - TRAIN XGBOOST MODEL
# -------------------------------------------------------------------------------------------------------------------------------------
print('Set XGBoost Parameters...')
num_rounds = 2975
params = {'eta': 0.01, 'max_depth': 18, 'colsample_bytree': 0.2, 'subsample': 0.8, 'colsample_bylevel':0.3, 'alpha':2,
            'objective': 'reg:linear', 'eval_metric': 'rmse', 'seed': 99, 'silent': True}

print('Building DMatrix...')
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(test)

print('Training XGBoost Model...')
gbdt = xgb.train(params, d_train, num_rounds, verbose_eval=10)

# -------------------------------------------------------------------------------------------------------------------------------------
# 										STEP 4 - GENERATE PREDICTIONS
# -------------------------------------------------------------------------------------------------------------------------------------
print('Prediction ...')
test_probs = gbdt.predict(d_test)
test_probs = np.expm1(test_probs)

# -------------------------------------------------------------------------------------------------------------------------------------
# 										STEP 5 - GENERATE KAGGLE SUBMISSION FILE
# -------------------------------------------------------------------------------------------------------------------------------------
print('Generate Submission ...')
predictions = predictions.append(pd.DataFrame({"id": test_pk, "visitors": test_probs}))
predictions.to_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Submissions\\XGB_Prediction.csv', index=False)
