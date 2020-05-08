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
    
    If any of the above pre-processing steps need to be excluded, set it to 'False' in Step 0
'''

# -------------------------------------------------------------------------------------------------------------------------------------
#          					IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------------------------
# 					STEP 0 - SET VARIABLES & WORKING DIRECTORY
# -------------------------------------------------------------------------------------------------------------------------------------

# Working directory location of the root folder
root_directory = 'C:\\GoogleDrive(UniMelb)\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting'

# Comma separated list of filename(s) with extensions which need to be pre-processed
preprocessing_files_list = ['train.csv']

# Specify preprocessing steps to be performed
remove_outlier = True

### Variables for outlier removal
outlier_file_list = ['train'] # Comma separated list of data file names (without extension) from which outliers need to be removed
outlier_column = 'visitors' # Name of the column which should be used to compute quartile range
lower_quartile = 25 # Any value in [0,100]. For computing quartile range
upper_quartile = 75 # Any value in [0,100]. For computing quartile range. Upper quartile should be higher than lower quartile
amplifier = 2 # In case quartile range needs to be amplified, else set to 1
keep_zero = True # If zero values are outside QR, do you still want to keep it?

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
# 							STEP 2 - PRE-PROCESSING
# -------------------------------------------------------------------------------------------------------------------------------------

def remove_outliers(data, outlier_column, k, keep_zero, lower_quartile, upper_quartile): 
    '''
    Remove outliers based on a numeric column whose values are outside of
    k * quartile_range where k is a tuning parameter
    
    Parameter description:            
        (1) outlier_data_file - Name of the data file (without extension) from which outlier needs to be removed
        (2) outlier_column - Name of the column which should be used to compute quartile range (QR)
        (3) k - In case QR needs to be amplified by a factor of k
        (4) keep_zero - If zero values are outside QR, do you still want to keep it?
		(5) lower_quartile - Any value in [0,100]. For computing quartile range
		(6) upper_quartile - Any value in [0,100]. For computing quartile range. Upper quartile should be higher than lower quartile
    '''      
    # Compute quartile range
    q1 = np.percentile(data[outlier_column], lower_quartile, axis=0)
    q3 = np.percentile(data[outlier_column], upper_quuartile, axis=0)
    qr = q3 - q1                               
    
    # Remove outliers
    lower_limit = q1 - amplifier*qr
    upper_limit = q3 + amplifier*qr
    df_qr_range = data.loc[data[outlier_column] > lower_limit]
    df_qr_range = df_qr_range.loc[df_qr_range[outlier_column] < q3 + k*qr]

    if keep_zero==True and not (lower_limit < 0 < upper_limit):
        df_0 = data.loc[data[outlier_column] == 0]       
        return pd.concat([df_0, df_qr_range])
    
    return df_qr_range
    
# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 3 - EXECUTE PRE-PROCESSING
# -------------------------------------------------------------------------------------------------------------------------------------

if remove_outlier == True:
    print('Removing outliers ...')
    for file in outlier_file_list:
        data[file] = remove_outliers(data[file], outlier_column, amplifier, keep_zero, lower_quartile, upper_quartile)

del file

# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 4 - SAVE PRE-PROCESSED DATA
# -------------------------------------------------------------------------------------------------------------------------------------

print('Saving preprocessed files ...')
for key in data:
    data[key].to_csv(data_folder + '\\' + key + '-preprocessed.csv', index=False)

del key, amplifier, keep_zero, outlier_column, outlier_file_list, preprocessing_files_list, remove_outlier




