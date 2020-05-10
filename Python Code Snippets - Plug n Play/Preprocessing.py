# -------------------------------------------------------------------------------------------------------------------------------------
# 								README INSTRUCTIONS
# -------------------------------------------------------------------------------------------------------------------------------------
'''
(1) Folder structure should be created manually as follows:
       Root folder (named as <Competition Name>)
           /Data
           /Code
           
(2) Filenames should not have spaces in them

(3) All the variables should be set up correctly in STEP 0

(4) Do not change variables being set up in any other step(s)

(5) Following pre-processing steps can be performed:
		- Remove outliers based on inter-quartile range
        - Log transformation
        - Integer encoding
		- One-hot encoding
	If any of the above pre-processing steps need to be excluded, set it to 'False' in Step 0
'''

# -------------------------------------------------------------------------------------------------------------------------------------
#          					IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import to_categorical

# -------------------------------------------------------------------------------------------------------------------------------------
# 					STEP 0 - SET VARIABLES & WORKING DIRECTORY
# -------------------------------------------------------------------------------------------------------------------------------------

# Working directory location of the root folder
root_directory = 'C:\\GoogleDrive(UniMelb)\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting'

# Comma separated list of filename(s) with extensions which need to be pre-processed
preprocessing_files_list = ['train.csv']

# Specify preprocessing steps to be performed
remove_outlier     = True
log_transformation = True
integer_encoding   = True
one_hot_encoding   = True

### Variables for outlier removal
outlier_file_list  = ['train']    # Comma separated list of data file(s) name (without extension) from which outliers need to be removed
outlier_column     = 'visitors'   # Name of the column which should be used to compute quartile range. Should be numeric
lower_quartile     = 25           # Any value in [0,100]. For computing quartile range
upper_quartile     = 75           # Any value in [0,100]. For computing quartile range. Upper quartile should be higher than lower quartile
amplifier          = 2            # In case quartile range needs to be amplified, else set to 1
keep_zero          = True         # If zero values are outside QR, do you still want to keep it?

### Variables for log transformation
log_file_list      = ['train']    # Comma separated list of data file(s) name (without extension) requiring log transformation
log_columns        = ['visitors'] # Comma separated list column(s) on which log transformation needs to be done. Should be numeric

### Variables for integer encoding
int_encode_file_list  = ['train']    # Comma separated list of data file(s) name (without extension) requiring integer encoding
int_encode_columns    = ['air_area_name_mod', 'air_genre_name_mod'] # Comma separated list column(s) which need to be integer encoded. Should be categorical

### Variables for one-hot encoding
one_hot_file_list  = ['train']    # Comma separated list of data file(s) name (without extension) requiring one-hot encoding
one_hot_columns    = ['air_genre_name_mod'] # Comma separated list of column(s) which need to be one-hot encoded. Should be categorical

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

def integer_encode(data, columns):
	'''
	Converts categorical columns to numeric. Suitable where there is a natural ordinal relationship between the categories such as labels for temperature 'cold', 'warm', and 'hot'.
    
    Parameter description:
		(1) data 	- Dataframe for which column(s) need to be integer encoded
		(2) columns - List of categorical column(s) to integer encode
	'''
	
	for col in columns:
		lbl = preprocessing.LabelEncoder()
		data[col] = lbl.fit_transform(data[col].astype(str))  
		
	return data	

def one_hot_encode(data, columns):
	'''
	A one hot encoding is a representation of categorical variables as binary vectors.
    
    Parameter description:
		(1) data 	- Dataframe for which column(s) need to be one-hot encoded
		(2) columns - List of categorical column(s) to hot encode
	'''
    
    # Perform integer encoding first
	data = integer_encode(data, columns)
    
    # Perform one-hot encoding on integer encoded columns
	for col in columns:
		encoded = to_categorical(data[col])         # One-hot
		encoded = pd.DataFrame(encoded).astype(int) # Convert result to dataframe		
		encoded = encoded.add_prefix(col + '_')     # Add prefix to dummy columns
		data    = pd.concat([data, encoded], axis=1)
		data    = data.drop(columns = col)
		
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
    for file in int_encode_file_list:
        data[file] = integer_encode(data[file], int_encode_columns)
        
if one_hot_encoding == True:
    print('One-Hot encoding ...')
    for file in one_hot_file_list:
        data[file] = one_hot_encode(data[file], one_hot_columns)
	
del file

# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 4 - SAVE PREPROCESSED DATA
# -------------------------------------------------------------------------------------------------------------------------------------

print('Saving preprocessed files ...')
for key in data:
    data[key].to_csv(data_folder + '\\' + key + '-preprocessed.csv', index=False)

del key, amplifier, keep_zero, outlier_column, outlier_file_list, preprocessing_files_list, remove_outlier, log_transformation, log_columns, log_file_list, int_encode_file_list, int_encode_columns, one_hot_file_list, one_hot_columns


