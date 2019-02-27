import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn import preprocessing
from sklearn.cluster import KMeans

# -------------------------------------------------------------------------------------------------------------------------------------
# 												STEP 0 - READ DATA FILES
# -------------------------------------------------------------------------------------------------------------------------------------
data = {
    'tra': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\air_visit_data.csv'),
    'as': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\air_store_info.csv'),
    'hs': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\hpg_store_info.csv'),
    'ar': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\air_reserve.csv'),
    'hr': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\hpg_reserve.csv'),
    'id': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\store_id_relation.csv'),
    'tes': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\sample_submission.csv'),
    'hol': pd.read_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

# -------------------------------------------------------------------------------------------------------------------------------------
# 												STEP 1 - PRE-PROCESSING
# -------------------------------------------------------------------------------------------------------------------------------------

# Split concatenated 'ID' into 'Store ID' and 'Visit Date' in Test data
data['tes']['air_store_id'] = data['tes']['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = data['tes']['id'].apply(lambda x: str(x).split('_')[2])
data['tes'] = data['tes'].drop('id', axis=1)
data['tes'] = data['tes'].drop('visitors', axis=1)

# Log Transform Target Variable in Train data
data['tra'].visitors = np.log1p(data['tra'].visitors.values)

# Convert date columns to datetime
for df in ['tra', 'tes', 'hol']:
    data[df].visit_date = pd.to_datetime(data[df].visit_date)

# Convert datetime columns to date   
for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date

# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 2 - FEATURE ENGINEERING USING "DATE INFO" DATA
# -------------------------------------------------------------------------------------------------------------------------------------
   
# Create 'Year' and 'Month' features
data['hol']['year'] = data['hol']['visit_date'].dt.year
data['hol']['month_of_year'] = data['hol']['visit_date'].dt.month
    
# Create 'Is_Weekend' feature
data['hol']['is_weekend'] = 0
data['hol']['is_weekend'][data['hol'].day_of_week.isin(['Saturday', 'Sunday'])] = 1

# Holiday on weekend is not important
data['hol']['holiday_flg'][data['hol'].day_of_week.isin(['Saturday', 'Sunday'])] = 0
 
# Create feature for Season according to months
data['hol']['season'] = 0
data['hol']['season'][data['hol'].month_of_year.isin([3,4,5])] = 1
data['hol']['season'][data['hol'].month_of_year.isin([6,7,8])] = 2
data['hol']['season'][data['hol'].month_of_year.isin([9,10,11])] = 3

# Create feature whether it is first, mid or last ten days of the month
data['hol']['is_first_mid_last_ten_days'] = 0
data['hol']['is_first_mid_last_ten_days'][(data['hol'].visit_date.dt.day > 10) & (data['hol'].visit_date.dt.day <= 20)] = 1
data['hol']['is_first_mid_last_ten_days'][data['hol'].visit_date.dt.day > 20] = 2

# Create feature whether it is the first or second half of the month
data['hol']['is_first_last_half_month'] = 0
data['hol']['is_first_last_half_month'][data['hol'].visit_date.dt.day > 15] = 1

# Convert Factors to Numeric
lbl = preprocessing.LabelEncoder()
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
   
# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 3 - FEATURE ENGINEERING USING "STORE INFO" DATA
# -------------------------------------------------------------------------------------------------------------------------------------

# Create 'longitude minus latitude' feature
data['as']['lon_minus_lat'] = data['as']['longitude'] - data['as']['latitude'] 

# Create 'Fu' area name feature
data['as']['air_area_name_split'] = pd.DataFrame(data['as'].air_area_name.str.split(' ', 2))
data['as'][['area1','area2', 'area3']] = pd.DataFrame(data['as'].air_area_name_split.values.tolist())
data['as']['air_area_name_fu'] = data['as'].area1.str.partition("-fu")[0]
data['as']['air_area_name_fu'][data['as'].area1.str.find('-fu') == -1] = np.NaN
data['as']['air_area_name_fu'][data['as'].air_area_name_fu.isnull()] = data['as'].area2.str.partition("-fu")[0]
data['as']['air_area_name_fu'][(data['as'].area1.str.find('-fu') == -1) & (data['as'].area2.str.find('-fu') == -1)] = np.NaN
data['as']['air_area_name_fu'][data['as'].air_area_name_fu.isnull()] = data['as'].area3.str.partition("-fu")[0]
data['as']['air_area_name_fu'][(data['as'].area1.str.find('-fu') == -1) & (data['as'].area2.str.find('-fu') == -1) & (data['as'].area3.str.find('-fu') == -1)] = np.NaN
data['as'] = data['as'].drop(['air_area_name_split', 'area1', 'area3'], axis=1)

# Create feature for number of distinct Store Ids in an Area
num_of_stores_in_area = data['as'].groupby('air_area_name')['air_store_id'].nunique().reset_index()
num_of_stores_in_area = num_of_stores_in_area.rename(columns={'air_store_id': 'num_of_stores_in_area'})
data['as'] = pd.merge(data['as'], num_of_stores_in_area, how='left', on='air_area_name')

# Create feature for number of distinct Store Ids for each Genre
num_of_stores_in_genre = data['as'].groupby('air_genre_name')['air_store_id'].nunique().reset_index()
num_of_stores_in_genre = num_of_stores_in_genre.rename(columns={'air_store_id': 'num_of_stores_in_genre'})
data['as'] = pd.merge(data['as'], num_of_stores_in_genre, how='left', on='air_genre_name')

# Create feature for number of distinct Area names for each Genre
num_of_areas_in_genre = data['as'].groupby('air_genre_name')['air_area_name'].nunique().reset_index()
num_of_areas_in_genre = num_of_areas_in_genre.rename(columns={'air_area_name': 'num_of_areas_in_genre'})
data['as'] = pd.merge(data['as'], num_of_areas_in_genre, how='left', on='air_genre_name')

# Create feature for number of distinct Genres in each Area
num_of_genres_in_area = data['as'].groupby('air_area_name')['air_genre_name'].nunique().reset_index()
num_of_genres_in_area = num_of_genres_in_area.rename(columns={'air_genre_name': 'num_of_genres_in_area'})
data['as'] = pd.merge(data['as'], num_of_genres_in_area, how='left', on='air_area_name')

# Create feature for number of distinct Store Ids for each Genre in each Area
num_of_stores_in_area_genre = data['as'].groupby(['air_area_name','air_genre_name'])['air_store_id'].nunique().reset_index()
num_of_stores_in_area_genre = num_of_stores_in_area_genre.rename(columns={'air_store_id': 'num_of_stores_in_area_genre'})
data['as'] = pd.merge(data['as'], num_of_stores_in_area_genre, how='left', on=['air_area_name','air_genre_name'])

# Create feature by clustering stores into 5 groups based on longitude and latitude
kmeans = KMeans(n_clusters=5, random_state=0).fit(data['as'][['longitude','latitude']])
data['as']['cluster'] = kmeans.predict(data['as'][['longitude','latitude']])

# Split area and genre names on spaces into separate features while converting factors to numeric
data['as']['air_genre_name_mod'] = data['as']['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
data['as']['air_area_name_mod'] = data['as']['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

lbl = preprocessing.LabelEncoder()
for i in range(10):
    data['as']['air_genre_name'+str(i)] = lbl.fit_transform(data['as']['air_genre_name_mod'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    data['as']['air_area_name'+str(i)] = lbl.fit_transform(data['as']['air_area_name_mod'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

# Convert Factors to Numeric
for cols in ['area2', 'air_genre_name', 'air_area_name', 'air_area_name_fu']:
    lbl = preprocessing.LabelEncoder()
    data['as'][cols] = lbl.fit_transform(data['as'][cols].astype(str))  

lbl = preprocessing.LabelEncoder()
lbl_fit = lbl.fit(pd.DataFrame(pd.concat([data['tra']['air_store_id'], data['tes']['air_store_id']])))

for df in ['tra', 'tes']:
    data[df]['air_store_id_2'] = lbl_fit.transform(data[df]['air_store_id'])
      
# Merge "Store Info" and "Date Info" Features with Train and Test Data
# Train data
data['tra'] = pd.merge(data['tra'], data['as'], how='left', on = 'air_store_id')
data['tra'] = pd.merge(data['tra'], data['hol'], how='left', left_on='visit_date', right_on='visit_date')

# Test data
data['tes'] = pd.merge(data['tes'], data['as'], how='left', on = 'air_store_id')
data['tes'] = pd.merge(data['tes'], data['hol'], how='left', left_on='visit_date', right_on='visit_date')

# ------------------------------------------------------------------------------------------------------------------------------------
# 							STEP 2 - REMOVE OUTLIERS FROM VISIT DATA (TRAIN)
# ------------------------------------------------------------------------------------------------------------------------------------

def remove_outliers(data):
    df_0 = data.loc[data.visitors == 0]   
    q1 = np.percentile(data.visitors, 25, axis=0)
    q3 = np.percentile(data.visitors, 75, axis=0)
    k = 4
    iqr = q3 - q1
    df_temp = data.loc[data.visitors > q1 - k*iqr]
    df_temp = data.loc[data.visitors < q3 + k*iqr]
    frames = [df_0, df_temp]
    result = pd.concat(frames)
    return result

data['tra'] = remove_outliers(data['tra'])

# -----------------------------------------------------------------------------------------------------------------------------------
# 							STEP 3 - FEATURE ENGINEERING USING "VISITORS INFO"
# 		(FEATURES TO CATEGORIZE POPULAR STORES, AREAS, AND GENRES ACCORDING TO NUMBER OF VISITORS)
# -----------------------------------------------------------------------------------------------------------------------------------
    
# Create feature for average number of visitors for each store 
store_visitor_size = data['tra'].groupby(['air_store_id'])['visitors'].mean().to_frame().reset_index()
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], store_visitor_size, how='left', on=['air_store_id'])
data['tra'] = data['tra'].rename(columns={'visitors_x': 'visitors', 'visitors_y': 'store_size'})
data['tes'] = data['tes'].rename(columns={'visitors': 'store_size'})

# Create feature by grouping stores into buckets based on their average number of visitors
store_visitor_size['size_grp_store'] = 1
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 5) & (store_visitor_size.visitors.values <= 10)] = 2
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 10) & (store_visitor_size.visitors.values <= 15)] = 3
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 15) & (store_visitor_size.visitors.values <= 20)] = 4
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 20) & (store_visitor_size.visitors.values <= 25)] = 5
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 25) & (store_visitor_size.visitors.values <= 30)] = 6
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 30) & (store_visitor_size.visitors.values <= 35)] = 7
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 35) & (store_visitor_size.visitors.values <= 40)] = 8
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 40) & (store_visitor_size.visitors.values <= 45)] = 9
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 45) & (store_visitor_size.visitors.values <= 50)] = 10
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 50) & (store_visitor_size.visitors.values <= 55)] = 11
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 55) & (store_visitor_size.visitors.values <= 60)] = 12
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 60) & (store_visitor_size.visitors.values <= 65)] = 13
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 65) & (store_visitor_size.visitors.values <= 70)] = 14
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 70) & (store_visitor_size.visitors.values <= 75)] = 15
store_visitor_size['size_grp_store'][(store_visitor_size.visitors.values > 75) & (store_visitor_size.visitors.values <= 80)] = 16
store_visitor_size = store_visitor_size.drop('visitors', axis=1)
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], store_visitor_size, how='left', on=['air_store_id'])

# Create feature for average number of visitors in each area across alll the stores in that area 
store_visitor_size = data['tra'].groupby(['air_area_name', 'visit_date'])['visitors'].sum().to_frame().reset_index()
store_visitor_size = store_visitor_size.groupby(['air_area_name'])['visitors'].mean().to_frame().reset_index()

# Create feature by grouping areas into buckets based on their average number of visitors
store_visitor_size['size_grp_area'] = 1
store_visitor_size['size_grp_area'][(store_visitor_size.visitors.values > 150) & (store_visitor_size.visitors.values <= 300)] = 2
store_visitor_size['size_grp_area'][(store_visitor_size.visitors.values > 300) & (store_visitor_size.visitors.values <= 450)] = 3
store_visitor_size['size_grp_area'][(store_visitor_size.visitors.values > 450) & (store_visitor_size.visitors.values <= 600)] = 4
store_visitor_size['size_grp_area'][(store_visitor_size.visitors.values > 600) & (store_visitor_size.visitors.values <= 750)] = 5
store_visitor_size['size_grp_area'][(store_visitor_size.visitors.values > 750) & (store_visitor_size.visitors.values <= 900)] = 6
store_visitor_size = store_visitor_size.drop('visitors', axis=1)
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], store_visitor_size, how='left', on=['air_area_name'])

# Create feature for average number of visitors for each genre across alll the stores in that genre 
store_visitor_size = data['tra'].groupby(['air_genre_name', 'visit_date'])['visitors'].sum().to_frame().reset_index()
store_visitor_size = store_visitor_size.groupby(['air_genre_name'])['visitors'].mean().to_frame().reset_index()
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], store_visitor_size, how='left', on=['air_genre_name'])
data['tra'] = data['tra'].rename(columns={'visitors_x': 'visitors', 'visitors_y': 'genre_size'})
data['tes'] = data['tes'].rename(columns={'visitors': 'genre_size'})

# -------------------------------------------------------------------------------------------------------------------------------------
# STEP 4 - FEATURE ENGINEERING USING STATISTICAL CALCULATIONS ON NUMBER OF VISITORS ACROSS COMBINATION OF MULTIPLE DIFFERENT COLUMNS
# -------------------------------------------------------------------------------------------------------------------------------------

# minimum, maximum, median, mean visitors by store and day of week
tmp = data['tra'].groupby(['air_store_id', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'day_of_week', 'min_visitors_store_dow', 'mean_visitors_store_dow', 'median_visitors_store_dow', 'max_visitors_store_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id', 'day_of_week'])

# minimum, maximum, median, mean visitors by air_genre_name and day of week
tmp = data['tra'].groupby(['air_genre_name', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'day_of_week', 'min_visitors_genre_dow', 'mean_visitors_genre_dow', 'median_visitors_genre_dow','max_visitors_genre_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name', 'day_of_week'])

# minimum, maximum, median, mean visitors by air_area_name and day of week
tmp = data['tra'].groupby(['air_area_name', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'day_of_week', 'min_visitors_area_dow', 'mean_visitors_area_dow', 'median_visitors_area_dow','max_visitors_area_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'day_of_week'])

# minimum, maximum, median, mean visitors by air_area_name, air_genre_name, and day of week
tmp = data['tra'].groupby(['air_area_name', 'air_genre_name', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'day_of_week', 'min_visitors_area_genre_dow', 'mean_visitors_area_genre_dow', 'median_visitors_area_genre_dow', 'max_visitors_area_genre_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name', 'day_of_week'])

# minimum, maximum, median, mean visitors by air_store_id, and is_weekend
tmp = data['tra'].groupby(['air_store_id', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'is_weekend', 'min_visitors_store_weekend_weekday', 'mean_visitors_store_weekend_weekday', 'median_visitors_store_weekend_weekday', 'max_visitors_store_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id', 'is_weekend'])

# minimum, maximum, median, mean visitors by genre, and is_weekend
tmp = data['tra'].groupby(['air_genre_name', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'is_weekend', 'min_visitors_genre_weekend_weekday', 'mean_visitors_genre_weekend_weekday', 'median_visitors_genre_weekend_weekday', 'max_visitors_genre_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name', 'is_weekend'])

# minimum, maximum, median, mean visitors by area, and is_weekend
tmp = data['tra'].groupby(['air_area_name', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'is_weekend', 'min_visitors_area_weekend_weekday', 'mean_visitors_area_weekend_weekday', 'median_visitors_area_weekend_weekday', 'max_visitors_area_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'is_weekend'])

# minimum, maximum, median, mean visitors by area, genre, and is_weekend
tmp = data['tra'].groupby(['air_area_name', 'air_genre_name', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'is_weekend', 'min_visitors_area_genre_weekend_weekday', 'mean_visitors_area_genre_weekend_weekday', 'median_visitors_area_genre_weekend_weekday', 'max_visitors_area_genre_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name', 'is_weekend'])

# minimum, maximum, median, mean visitors by store, holiday, and day_of_week
tmp = data['tra'].groupby(['air_store_id', 'holiday_flg', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'holiday_flg', 'day_of_week', 'min_visitors_store_holiday_dow', 'mean_visitors_store_holiday_dow', 'median_visitors_store_holiday_dow', 'max_visitors_store_holiday_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id', 'holiday_flg', 'day_of_week'])

# minimum, maximum, median, mean visitors by genre, holiday, and day_of_week
tmp = data['tra'].groupby(['air_genre_name', 'holiday_flg', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'holiday_flg', 'day_of_week', 'min_visitors_genre_holiday_dow', 'mean_visitors_genre_holiday_dow', 'median_visitors_genre_holiday_dow', 'max_visitors_genre_holiday_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name', 'holiday_flg', 'day_of_week'])

# minimum, maximum, median, mean visitors by genre, holiday, and day_of_week
tmp = data['tra'].groupby(['air_area_name', 'holiday_flg', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'holiday_flg', 'day_of_week', 'min_visitors_area_holiday_dow', 'mean_visitors_area_holiday_dow', 'median_visitors_area_holiday_dow', 'max_visitors_area_holiday_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'holiday_flg', 'day_of_week'])

# minimum, maximum, median, mean visitors by area, genre, holiday, and day_of_week
tmp = data['tra'].groupby(['air_area_name', 'air_genre_name', 'holiday_flg', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'holiday_flg', 'day_of_week', 'min_visitors_area_genre_holiday_dow', 'mean_visitors_area_genre_holiday_dow', 'median_visitors_area_genre_holiday_dow', 'max_visitors_area_genre_holiday_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name', 'holiday_flg', 'day_of_week'])

# minimum, maximum, median, mean visitors by air_store_id, holiday, and is_weekend
tmp = data['tra'].groupby(['air_store_id', 'holiday_flg', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'holiday_flg', 'is_weekend', 'min_visitors_store_holiday_weekend_weekday', 'mean_visitors_store_holiday_weekend_weekday', 'median_visitors_store_holiday_weekend_weekday', 'max_visitors_store_holiday_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id', 'holiday_flg', 'is_weekend'])

# minimum, maximum, median, mean visitors by genre, holiday, and is_weekend
tmp = data['tra'].groupby(['air_genre_name', 'holiday_flg', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'holiday_flg', 'is_weekend', 'min_visitors_genre_holiday_weekend_weekday', 'mean_visitors_genre_holiday_weekend_weekday', 'median_visitors_genre_holiday_weekend_weekday', 'max_visitors_genre_holiday_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name', 'holiday_flg', 'is_weekend'])

# minimum, maximum, median, mean visitors by area, holiday, and is_weekend
tmp = data['tra'].groupby(['air_area_name', 'holiday_flg', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'holiday_flg', 'is_weekend', 'min_visitors_area_holiday_weekend_weekday', 'mean_visitors_area_holiday_weekend_weekday', 'median_visitors_area_holiday_weekend_weekday', 'max_visitors_area_holiday_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'holiday_flg', 'is_weekend'])

# minimum, maximum, median, mean visitors by area, genre, holiday, and is_weekend
tmp = data['tra'].groupby(['air_area_name', 'air_genre_name', 'holiday_flg', 'is_weekend']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'holiday_flg', 'is_weekend', 'min_visitors_area_genre_holiday_weekend_weekday', 'mean_visitors_area_genre_holiday_weekend_weekday', 'median_visitors_area_genre_holiday_weekend_weekday', 'max_visitors_area_genre_holiday_weekend_weekday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name', 'holiday_flg', 'is_weekend'])

# minimum, maximum, median, mean visitors by area, and holiday
tmp = data['tra'].groupby(['air_area_name', 'holiday_flg']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'holiday_flg', 'min_visitors_area_holiday', 'mean_visitors_area_holiday', 'median_visitors_area_holiday', 'max_visitors_area_holiday']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'holiday_flg'])

# minimum, maximum, median, mean visitors by area, genre, and holiday
tmp = data['tra'].groupby(['air_area_name', 'air_genre_name', 'holiday_flg']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'holiday_flg', 'min_visitors_area_genre_holiday_flg', 'mean_visitors_area_genre_holiday_flg', 'median_visitors_area_genre_holiday_flg', 'max_visitors_area_genre_holiday_flg']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name', 'holiday_flg'])

# minimum, maximum, median, mean visitors in first ten days of month by store
tmp = data['tra'][data['tra'].visit_date.dt.day <= 10].groupby(['air_store_id']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'min_visitors_store_first_ten_days_of_month', 'mean_visitors_store_first_ten_days_of_month', 'median_visitors_store_first_ten_days_of_month', 'max_visitors_store_first_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id'])

# minimum, maximum, median, mean visitors in mid ten days of month by store
tmp = data['tra'][(data['tra'].visit_date.dt.day > 10) & (data['tra'].visit_date.dt.day <= 20)].groupby(['air_store_id']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'min_visitors_store_mid_ten_days_of_month', 'mean_visitors_store_mid_ten_days_of_month', 'median_visitors_store_mid_ten_days_of_month', 'max_visitors_store_mid_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id'])

# minimum, maximum, median, mean visitors in last ten days of month by store
tmp = data['tra'][data['tra'].visit_date.dt.day > 20].groupby(['air_store_id']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'min_visitors_store_last_ten_days_of_month', 'mean_visitors_store_last_ten_days_of_month', 'median_visitors_store_last_ten_days_of_month', 'max_visitors_store_last_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id'])

# minimum, maximum, median, mean visitors in first 15 days of month by store
tmp = data['tra'][data['tra'].visit_date.dt.day <= 15].groupby(['air_store_id']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'min_visitors_store_first_15_days_of_month', 'mean_visitors_store_first_15_days_of_month', 'median_visitors_store_first_15_days_of_month', 'max_visitors_store_first_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id'])

# minimum, maximum, median, mean visitors in last 15 days of month by store
tmp = data['tra'][data['tra'].visit_date.dt.day > 15].groupby(['air_store_id']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_store_id', 'min_visitors_store_last_15_days_of_month', 'mean_visitors_store_last_15_days_of_month', 'median_visitors_store_last_15_days_of_month', 'max_visitors_store_last_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_store_id'])

# minimum, maximum, median, mean visitors in first ten days of month by genre
tmp = data['tra'][data['tra'].visit_date.dt.day < 10].groupby(['air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'min_visitors_genre_first_ten_days_of_month', 'mean_visitors_genre_first_ten_days_of_month', 'median_visitors_genre_first_ten_days_of_month', 'max_visitors_genre_first_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name'])

# minimum, maximum, median, mean visitors in mid ten days of month by genre
tmp = data['tra'][(data['tra'].visit_date.dt.day >= 10) & (data['tra'].visit_date.dt.day < 20)].groupby(['air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'min_visitors_genre_mid_ten_days_of_month', 'mean_visitors_genre_mid_ten_days_of_month', 'median_visitors_genre_mid_ten_days_of_month', 'max_visitors_genre_mid_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name'])

# minimum, maximum, median, mean visitors in last ten days of month by genre
tmp = data['tra'][data['tra'].visit_date.dt.day >= 20].groupby(['air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'min_visitors_genre_last_ten_days_of_month', 'mean_visitors_genre_last_ten_days_of_month', 'median_visitors_genre_last_ten_days_of_month', 'max_visitors_genre_last_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name'])

# minimum, maximum, median, mean visitors in first 15 days of month by genre
tmp = data['tra'][data['tra'].visit_date.dt.day <= 15].groupby(['air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'min_visitors_genre_first_15_days_of_month', 'mean_visitors_genre_first_15_days_of_month', 'median_visitors_genre_first_15_days_of_month', 'max_visitors_genre_first_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name'])

# minimum, maximum, median, mean visitors in last 15 days of month by genre
tmp = data['tra'][data['tra'].visit_date.dt.day > 15].groupby(['air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_genre_name', 'min_visitors_genre_last_15_days_of_month', 'mean_visitors_genre_last_15_days_of_month', 'median_visitors_genre_last_15_days_of_month', 'max_visitors_genre_last_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_genre_name'])

# minimum, maximum, median, mean visitors in first ten days of month by area
tmp = data['tra'][data['tra'].visit_date.dt.day < 10].groupby(['air_area_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'min_visitors_area_first_ten_days_of_month', 'mean_visitors_area_first_ten_days_of_month', 'median_visitors_area_first_ten_days_of_month', 'max_visitors_area_first_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name'])

# minimum, maximum, median, mean visitors in mid ten days of month by area
tmp = data['tra'][(data['tra'].visit_date.dt.day >= 10) & (data['tra'].visit_date.dt.day < 20)].groupby(['air_area_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'min_visitors_area_mid_ten_days_of_month', 'mean_visitors_area_mid_ten_days_of_month', 'median_visitors_area_mid_ten_days_of_month', 'max_visitors_area_mid_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name'])

# minimum, maximum, median, mean visitors in last ten days of month by area
tmp = data['tra'][data['tra'].visit_date.dt.day >= 20].groupby(['air_area_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'min_visitors_area_last_ten_days_of_month', 'mean_visitors_area_last_ten_days_of_month', 'median_visitors_area_last_ten_days_of_month', 'max_visitors_area_last_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name'])

# minimum, maximum, median, mean visitors in first 15 days of month by area
tmp = data['tra'][data['tra'].visit_date.dt.day <= 15].groupby(['air_area_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'min_visitors_area_first_15_days_of_month', 'mean_visitors_area_first_15_days_of_month', 'median_visitors_area_first_15_days_of_month', 'max_visitors_area_first_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name'])

# minimum, maximum, median, mean visitors in last 15 days of month by area
tmp = data['tra'][data['tra'].visit_date.dt.day > 15].groupby(['air_area_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'min_visitors_area_last_15_days_of_month', 'mean_visitors_area_last_15_days_of_month', 'median_visitors_area_last_15_days_of_month', 'max_visitors_area_last_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name'])

# minimum, maximum, median, mean visitors in first ten days of month by area, and genre
tmp = data['tra'][data['tra'].visit_date.dt.day < 10].groupby(['air_area_name', 'air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'min_visitors_genre_area_first_ten_days_of_month', 'mean_visitors_genre_area_first_ten_days_of_month', 'median_visitors_genre_area_first_ten_days_of_month', 'max_visitors_genre_area_first_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name'])

# minimum, maximum, median, mean visitors in mid ten days of month by area, and genre
tmp = data['tra'][(data['tra'].visit_date.dt.day >= 10) & (data['tra'].visit_date.dt.day < 20)].groupby(['air_area_name', 'air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'min_visitors_genre_area_mid_ten_days_of_month', 'mean_visitors_genre_area_mid_ten_days_of_month', 'median_visitors_genre_area_mid_ten_days_of_month', 'max_visitors_genre_area_mid_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name'])

# minimum, maximum, median, mean visitors in last ten days of month by area, and genre
tmp = data['tra'][data['tra'].visit_date.dt.day >= 20].groupby(['air_area_name', 'air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'min_visitors_genre_area_last_ten_days_of_month', 'mean_visitors_genre_area_last_ten_days_of_month', 'median_visitors_genre_area_last_ten_days_of_month', 'max_visitors_genre_area_last_ten_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name'])

# minimum, maximum, median, mean visitors in first 15 days of month by area, and genre
tmp = data['tra'][data['tra'].visit_date.dt.day <= 15].groupby(['air_area_name', 'air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'min_visitors_genre_area_first_15_days_of_month', 'mean_visitors_genre_area_first_15_days_of_month', 'median_visitors_genre_area_first_15_days_of_month', 'max_visitors_genre_area_first_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name'])

# minimum, maximum, median, mean visitors in last 15 days of month by area, and genre
tmp = data['tra'][data['tra'].visit_date.dt.day > 15].groupby(['air_area_name', 'air_genre_name']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['air_area_name', 'air_genre_name', 'min_visitors_genre_area_last_15_days_of_month', 'mean_visitors_genre_area_last_15_days_of_month', 'median_visitors_genre_area_last_15_days_of_month', 'max_visitors_genre_area_last_15_days_of_month']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['air_area_name', 'air_genre_name'])

# minimum, maximum, median, mean visitors by lon_minus_lat and day of week
tmp = data['tra'].groupby(['lon_minus_lat', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['lon_minus_lat', 'day_of_week', 'min_visitors_lat_plus_long_dow', 'mean_visitors_lat_plus_long_dow', 'median_visitors_lat_plus_long_dow', 'max_visitors_lat_plus_long_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['lon_minus_lat', 'day_of_week'])

# minimum, maximum, median, mean visitors by area2 and day of week
tmp = data['tra'].groupby(['area2', 'day_of_week']).agg({'visitors' : [np.min,np.mean,np.median,np.max]}).reset_index()
tmp.columns = ['area2', 'day_of_week', 'min_visitors_area2_dow', 'mean_visitors_area2_dow', 'median_visitors_area2_dow', 'max_visitors_area2_dow']
for df in ['tra','tes']:
    data[df] = pd.merge(data[df], tmp, how='left', on=['area2', 'day_of_week'])

# -------------------------------------------------------------------------------------------------------------------------------------
# 								       STEP 5 - FURTHER FEATURE ENGINEERING USING KAGGLE FORUMS
# -------------------------------------------------------------------------------------------------------------------------------------
# Get air_store_id
data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

# For each Store and Visit Date, create features for 
#    (1) Avg difference of days between reservation and visit dates 
#    (2) Avg number of visitors reserved
#    (3) Total difference of days between reservation and visit dates 
#    (4) Total number of visitors reserved
for df in ['ar','hr']:
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)   
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'total_reserve_datetime_diff', 'reserve_visitors':'total_reserved_visitors'})
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'avg_reserve_datetime_diff', 'reserve_visitors':'avg_reserved_visitors'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])
    data[df].visit_date = pd.to_datetime(data[df].visit_date)
    data['tra'] = pd.merge(data['tra'], data[df], how='left', on=['air_store_id','visit_date']) 
    data['tes'] = pd.merge(data['tes'], data[df], how='left', on=['air_store_id','visit_date'])

# From both the systems, for each store and visit_date
#   (1) How many visitors across all reservations
#   (2) How many average visitors per reservation
#   (3) On average per reservation, how may days in advance were the reservations made
#   (4) Year and month concatenated as integer
#   (5) Difference between latitude of the current store and maximum latitude across all stores
#   (6) Difference between longitude of the current store and maximum longitude across all stores
for df in ['tra','tes']:
    data[df]['total_reserved_visitors'] = data[df]['total_reserved_visitors_x'] + data[df]['total_reserved_visitors_y']
    data[df]['avg_reserved_visitors'] = (data[df]['avg_reserved_visitors_x'] + data[df]['avg_reserved_visitors_y']) / 2
    data[df]['avg_reserve_datetime_diff'] = (data[df]['avg_reserve_datetime_diff_x'] + data[df]['avg_reserve_datetime_diff_y']) / 2
    data[df]['date_int'] = data[df]['visit_date'].apply(lambda x: x.strftime('%Y%m')).astype(int)
    data[df]['var_max_lat'] = data[df]['latitude'].max() - data[df]['latitude']
    data[df]['var_max_long'] = data[df]['longitude'].max() - data[df]['longitude']

# Convert to date
for df in ['tra', 'tes', 'ar', 'hr']:
    data[df]['visit_date'] = data[df]['visit_date'].dt.date
    
# -------------------------------------------------------------------------------------------------------------------------------------
# 											STEP 6 - REPLACE MISSING VALUES WITH -1
# -------------------------------------------------------------------------------------------------------------------------------------
for df in ['tra','tes']:
    data[df] = data[df].fillna(-1)

del tmp1, tmp2, i, df, cols, num_of_stores_in_area, num_of_stores_in_genre, num_of_areas_in_genre, num_of_genres_in_area, num_of_stores_in_area_genre, tmp, store_visitor_size

# -------------------------------------------------------------------------------------------------------------------------------------
# 								STEP 7 - SAVE FEATURE ENGINEERED TRAIN AND TEST DATA SETS
# -------------------------------------------------------------------------------------------------------------------------------------
data['tra'].to_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\train.csv', index=False)
data['tes'].to_csv('C:\\Users\\Muhammad Atif\\Google Drive\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting\\Data\\test.csv', index=False)



