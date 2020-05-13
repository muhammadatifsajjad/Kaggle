# -------------------------------------------------------------------------------------------------------------------------------------
# 								README INSTRUCTIONS
# -------------------------------------------------------------------------------------------------------------------------------------
'''
(1) Folder structure should be created manually as follows:
       Root folder (named as <Competition Name>)
           /Data
           /Code
           
(2) Train and Test files should be in a csv format and named as follows :
		Train file name   		  - train.csv
		Validation/Test file name - test.csv

(3) All the variables should be set up correctly in STEP 0

(4) Do not change variables being set up in any other step(s)

(5) Following pre-processing steps can be performed:
		- Remove outliers based on inter-quartile range
        - Log transformation
        - Integer encoding
		- One-hot encoding
		- Null vaues imputation (With 0, placeholder, and median for numeric; mode for categorical)
        - Feature sclaing (Using min-max scaling, inter-quartile scaling, z-score standardization)
	If any of the above pre-processing steps need to be excluded, set it to 'False' in Step 0
'''

# -------------------------------------------------------------------------------------------------------------------------------------
#          					IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# -------------------------------------------------------------------------------------------------------------------------------------
# 					STEP 0 - SET VARIABLES & WORKING DIRECTORY
# -------------------------------------------------------------------------------------------------------------------------------------

# Working directory location of the root folder
root_directory = 'C:\\GoogleDrive(UniMelb)\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting'

# Comma separated list of filenames (with extensions) which need to be pre-processed. This should not be changed.
preprocessing_files_list = ['train.csv', 'test.csv']

# Specify preprocessing steps to be performed
remove_outlier     = True
log_transformation = True
integer_encoding   = True
one_hot_encoding   = True
null_imputation	   = True
feature_scaling	   = True

### Variables for outlier removal
outlier_file_list  = ['train']    # Comma separated list of data file(s) name (without extension) from which outliers need to be removed
outlier_column     = 'visitors'   # Name of the numeric column which should be used to compute quartile range
lower_quartile     = 25           # Any value in [0,100]. For computing quartile range
upper_quartile     = 75           # Any value in [0,100]. For computing quartile range. Upper quartile should be higher than lower quartile
amplifier          = 2            # In case quartile range needs to be amplified, else set to 1
keep_zero          = True         # If zero values are outside QR, do you still want to keep it?

### Variables for log transformation
log_file_list = ['train']    # Comma separated list of data file(s) name (without extension) requiring log transformation
log_columns   = ['visitors'] # Comma separated list of positive numeric column(s) on which log transformation needs to be done

### Variables for integer encoding
int_encode_file_list  = ['train', 'test'] # Comma separated list of data file(s) name (without extension) requiring integer encoding. This should not be changed
int_encode_columns    = ['Bridge_Types'] # Comma separated list of categorical  column(s) which need to be integer encoded
default_value_int_enc = 'Cantilever'		# If there are unseen values in the test file, what default value should it be mapped to. One option can be to use the most frequent category

### Variables for one-hot encoding
one_hot_file_list     = ['train', 'test'] # Comma separated list of data file(s) name (without extension) requiring one-hot encoding. This should not be changed
one_hot_columns       = ['Bridge_Types'] # Comma separated list of categorical column(s) which need to be one-hot encoded
default_value_one_hot = 'Cantilever'	# If there are unseen values in the test file, what default value should it be mapped to while doing integer encoding. One option can be to use the most frequent category

### Variables for null values imputation
null_impute_file_list 				 = ['train', 'test'] # Comma separated list of data file(s) name (without extension) requiring null values imputation. This should not be changed
null_impute_columns_with_0	  		 = [] # Comma separated list of numeric column(s) which need to be imputed with 0
null_impute_columns_with_placeholder = [] # Comma separated list of numeric/categorical column(s) which need to be imputed with a placeholder value
placeholder_value 					 = -999 # Numeric/String placeholder value with which to replace missing values in the above specified columns
null_impute_columns_with_median 	 = [] # Comma separated list of numeric column(s) which need to be imputed with median
null_impute_columns_with_mode_cat 	 = ['x'] # Comma separated list of categorical column(s) which need to be imputed with the most dominant category

### Variables for feature scaling
feat_scale_file_list   		 = ['train', 'test'] # Comma separated list of data file(s) name (without extension) requiring feature scaling. This should not be changed
min_max_scale_columns  		 = ['x1'] # Comma separated list of numeric column(s) for which min-max normalization needs to be performed
inter_quartile_scale_columns = [] # Comma separated list of numeric column(s) for which inter-quartile scaling needs to be performed
z_score_scale_columns 		 = ['x3'] # Comma separated list of numeric column(s) for which z-score scaling needs to be performed

# -------------------------------------------------------------------------------------------------------------------------------------
# 							STEP 1 - READ DATA FILES
# -------------------------------------------------------------------------------------------------------------------------------------

# Change working directory
os.chdir(root_directory)

# Read all the data files in a dictionary
data_folder = root_directory + "\\Data\\A"
data = {}
for subdir, dirs, files in os.walk(data_folder):
    for file in files:
        if file in preprocessing_files_list:
            data[os.path.splitext(file)[0]] = pd.read_csv(data_folder + "\\" + file)

del subdir, dirs, file, files    
        
# -------------------------------------------------------------------------------------------------------------------------------------
# 							STEP 2 - PREPROCESSING
# -------------------------------------------------------------------------------------------------------------------------------------

def remove_outliers(data, outlier_column, k, keep_zero, lower_quartile, upper_quartile): 
    '''
    Remove outliers based on a numeric column whose values are outside of
    k * quartile_range where k is a tuning parameter
    
    Parameter description:            
		(1) data 		   - Dataframe from which outliers need to be removed
		(2) outlier_column - Name of the numeric column which should be used to compute quartile range (QR)
		(3) k 			   - In case QR needs to be amplified by a factor of k
		(4) keep_zero 	   - If zero values are outside QR, do you still want to keep it?
		(5) lower_quartile - Any value in [0,100]. For computing quartile range
		(6) upper_quartile - Any value in [0,100]. For computing quartile range. Upper quartile should be higher than lower quartile
    '''      
    # Compute quartile range
    q1 = np.percentile(data[outlier_column], lower_quartile, axis=0)
    q3 = np.percentile(data[outlier_column], upper_quartile, axis=0)
    qr = q3 - q1                               
    
    # Remove outliers
    lower_limit = q1 - amplifier*qr
    upper_limit = q3 + amplifier*qr
    df_qr_range = data[(data[outlier_column] > lower_limit) & (data[outlier_column] < upper_limit)]

    if keep_zero==True and not (lower_limit < 0 < upper_limit):
        df_0 = data.loc[data[outlier_column] == 0]       
        return pd.concat([df_0, df_qr_range])
    
    return df_qr_range

def log_transform(data, columns):
	'''
	Logarithmic transformation of numeric column. Columns must have all positive
    values because log is undefined for negative values. Alternatively, perform
    standardizing before applying log transform to column with negative values.
    
    Benefits:
		(1) It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
		(2) In most of the cases the magnitude order of the data changes within the range of the data. For instance, the difference between ages 15 and 20 is not equal to the ages 65 and 70. In terms of years, yes, they are identical, but for all other aspects, 5 years of difference in young ages mean a higher magnitude difference. This type of data comes from a multiplicative process and log transform normalizes the magnitude differences like that.
		(3) It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.
   
    Parameter description:
		(1) data 	- Dataframe for which column(s) need to be log transformed
		(2) columns - List of numeric column(s) to log transform
	'''
    
	for col in columns:
		data[col] = np.log1p(data[col])
		
	return data

def integer_encode(data, columns, default_value):
	'''
	Converts categorical columns to numeric. Suitable where there is a natural ordinal 
	relationship between the categories such as labels for temperature 'cold', 'warm', and 'hot'.
    
	Parameter description:
		(1) data 		  - Dataframe for which column(s) need to be integer encoded
		(2) columns 	  - List of categorical column(s) to integer encode
		(3) default_value - If there are unseen values in the test file, what default value 
							should it be mapped to. One option can be to use the most frequent category
	'''
	
	for col in columns:
		le = preprocessing.LabelEncoder()
		data['train'][col] = le.fit_transform(data['train'][col])
		dic = dict(zip(le.classes_, le.transform(le.classes_)))
		data['test'][col] = data['test'][col].map(dic).fillna(dic[default_value]).astype(int)

	return data	

def one_hot_encode(data, columns, default_value):
	'''
	A one hot encoding is a representation of categorical variables as binary vectors.
    
    Parameter description:
		(1) data 	- Dataframe for which column(s) need to be one-hot encoded
		(2) columns - List of categorical column(s) to hot encode
		(3) default_value - If there are unseen values in the test file, what default value 
					        should it be mapped to during integer_encoding. One option can be 
                            to use the most frequent category
	'''
    
    # Perform integer encoding first
	data = integer_encode(data, columns, default_value)
    
    # Perform one-hot encoding on integer encoded columns
	for col in columns:
		one_hot = preprocessing.OneHotEncoder()
		data['train'] = pd.concat([data['train'], pd.DataFrame(one_hot.fit_transform(data['train'][[col]]).toarray()).add_prefix(col + '_')], axis=1)
		data['test'] = pd.concat([data['test'], pd.DataFrame(one_hot.transform(data['test'][[col]]).toarray()).add_prefix(col + '_')], axis=1)

		# Drop original columns after being one-hot encoded
		for file in ['train', 'test']:
			data[file].drop(columns=col, inplace=True)		
	
	return data	

def impute_null_values(data, columns, method, placeholder_value):
	'''
	Null value imputation is used to fill missing values with an appropriate vaulue from 
	the data or domain knowledge
    
    Parameter description:
		(1) data 	- Dataframe for which column(s) need to be imputed
		(2) columns - List of categorical/numeric column(s) that need to be imputed
		(3) method  - Method to be used for imputation:
				(a) placeholder - For both numeric and categorical columns
				(b) 0 			- For numeric columns
				(c) median 		- For numeric columns
				(d) mode 		- For categorical columns
		(4) placeholder_value - Value to be used as a placeholder, if placeholder method is used
	'''
	for col in columns:
		if method == 'median':
		# Filling missing values with medians of the column in training data
			train_median = data['train'][col].median()
			data['train'][col].fillna(train_median, inplace=True) 
			data['test'][col].fillna(train_median, inplace=True) 
		if method == 'mode':		
		# Filling missing values with the most dominant category for categorical columns
			train_mode = data['train'][col].value_counts().idxmax()
			data['train'][col].fillna(train_mode, inplace=True) 
			data['test'][col].fillna(train_mode, inplace=True) 
		if method == 'placeholder':
		# Filling all missing values with the specified placeholder value
			data['train'][col].fillna(placeholder_value, inplace=True) 
			data['test'][col].fillna(placeholder_value, inplace=True) 
		if method == '0':
		# Filling all missing values with 0
			data['train'][col].fillna(0, inplace=True) 
			data['test'][col].fillna(0, inplace=True) 
		
	return data

def feature_scale(data, columns, method):
	'''
	Feature scaling to standardize features on a comparative scale range
    
    Parameter description:
		(1) data 	- Dataframe for which column(s) need to be scaled
		(2) columns - List of categorical/numeric column(s) that need to be scaled
		(3) method  - Method to be used for feature scaling:
				(a) placeholder   - For both numeric and categorical columns
				(b) 0 			  - For numeric columns
				(c) median 		  - For numeric columns
				(d) mode 		  - For categorical columns
	'''
    
	for col in columns:
		if method == 'min-max':
		# Perform min-max scaling
			scaler = preprocessing.MinMaxScaler()
			data['train'][col] = scaler.fit_transform(data['train'][[col]])
			data['test'][col]  = scaler.transform(data['test'][[col]])

		if method == 'inter-quartile':
		# Perform inter-quartile scaling
			scaler = preprocessing.RobustScaler()
			data['train'][col] = scaler.fit_transform(data['train'][[col]])
			data['test'][col]  = scaler.transform(data['test'][[col]])
			
		if method == 'z-score':	
		# Perform z-score standardization
			scaler = preprocessing.StandardScaler()
			data['train'][col] = scaler.fit_transform(data['train'][[col]])
			data['test'][col]  = scaler.transform(data['test'][[col]])
	
	return data

# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 3 - EXECUTE PREPROCESSING
# -------------------------------------------------------------------------------------------------------------------------------------

if remove_outlier == True:
	print('Removing outliers ...')
	for file in outlier_file_list:
		data[file] = remove_outliers(data[file], outlier_column, amplifier, keep_zero, lower_quartile, upper_quartile)
		data[file].reset_index(drop=True, inplace=True)
		
if log_transformation == True:
	print('Log Transformation ...')
	for file in log_file_list:
		data[file] = log_transform(data[file], log_columns)
		
if integer_encoding == True:
	print('Integer encoding ...')
	data = integer_encode(data, int_encode_columns, default_value_int_enc)
        
if one_hot_encoding == True:
	print('One-Hot encoding ...')
	data = one_hot_encode(data, one_hot_columns, default_value_one_hot)

if null_imputation == True:
	if null_impute_columns_with_0 != []:
		print('Null values imputation with 0 ...')
		data = impute_null_values(data, null_impute_columns_with_0, '0', None)
	if null_impute_columns_with_placeholder != []:
		print('Null values imputation with placeholder value ...')
		data = impute_null_values(data, null_impute_columns_with_placeholder, 'placeholder', placeholder_value)
	if null_impute_columns_with_median != []:
		print('Null values imputation with median ...')
		data = impute_null_values(data, null_impute_columns_with_median, 'median', None)
	if null_impute_columns_with_mode_cat != []:
		print('Null values imputation with mode category ...')
		data = impute_null_values(data, null_impute_columns_with_mode_cat, 'mode', None)

if feature_scaling == True:
	if min_max_scale_columns != []:
		print('Feature scaling using min-max scaler ...')
		data = feature_scale(data, min_max_scale_columns, 'min-max')	
	if inter_quartile_scale_columns != []:
		print('Feature scaling using inter-quartile scaler ...')
		data = feature_scale(data, inter_quartile_scale_columns, 'inter-quartile')	
	if z_score_scale_columns != []:
		print('Feature scaling using z-score scaler ...')
		data = feature_scale(data, z_score_scale_columns, 'z-score')	
			
del file

# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 4 - SAVE PREPROCESSED DATA
# -------------------------------------------------------------------------------------------------------------------------------------

print('Saving preprocessed files ...')
for key in data:
    data[key].to_csv(data_folder + '\\' + key + '-preprocessed.csv', index=False)

del key, amplifier, keep_zero, outlier_column, outlier_file_list, lower_quartile, upper_quartile, preprocessing_files_list, remove_outlier, log_transformation, log_columns, log_file_list, int_encode_file_list, int_encode_columns, one_hot_file_list, one_hot_columns, integer_encoding, one_hot_encoding, null_imputation, feature_scaling, default_value_int_enc, default_value_one_hot, null_impute_file_list, null_impute_columns_with_0, null_impute_columns_with_placeholder, placeholder_value, null_impute_columns_with_median, null_impute_columns_with_mode_cat, feat_scale_file_list, min_max_scale_columns, inter_quartile_scale_columns, z_score_scale_columns