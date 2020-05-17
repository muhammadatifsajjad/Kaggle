# -------------------------------------------------------------------------------------------------------------------------------------
# 								README INSTRUCTIONS
# -------------------------------------------------------------------------------------------------------------------------------------
'''
(1) Folder structure should be created manually as follows:
       Root folder (named as <Competition Name>)
           /Data
           /EDA
           /Submission
           /Code
           
(2) Train and Test files should be in a csv format named as 'train.csv'

(3) All the variables should be set up correctly in STEP 0

(4) Do not change variables being set up in any other step(s)

(5) Following pre-processing steps can be performed:
		- 
	If any of the above EDA steps need to be excluded, set it to 'False' in Step 0
'''

# -------------------------------------------------------------------------------------------------------------------------------------
#          					IMPORT REQUIRED PACKAGES
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from heatmap import heatmap, corrplot

# -------------------------------------------------------------------------------------------------------------------------------------
# 					STEP 0 - SET VARIABLES & WORKING DIRECTORY
# -------------------------------------------------------------------------------------------------------------------------------------

# Working directory location of the root folder
root_directory = 'C:\\GoogleDrive(UniMelb)\\Kaggle Competitions\\Recruit Restaurant Visitor Forecasting'

# Filename on which EDA needs to be done. This should not be changed.
eda_file_name = 'train'

# Specify EDA steps to be performed
data_types         = True
missing_data_stats = True
descriptive_stats  = True
correlation_plot   = True

# -------------------------------------------------------------------------------------------------------------------------------------
# 							STEP 1 - READ DATA FILES
# -------------------------------------------------------------------------------------------------------------------------------------

# Change working directory
os.chdir(root_directory)

# Read all the data files in a dictionary
# data_train = pd.read_csv(root_directory + "\\Data\\A\\" + eda_file_name + ".csv")
data_train = pd.read_csv('https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv')
data_train = data_train.replace('audi', np.nan)

# -------------------------------------------------------------------------------------------------------------------------------------
# 					STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------------------------------------------------------------------------

def miss_data_stats(data):
	'''
	Get null values count and percentage for all columns in the data
    
    Parameter description:
		(1) data - Dataframe for which missing data stats are needed
	'''

	missing_stats = pd.DataFrame(data.isna().sum())
	missing_stats.columns = ['Null Count']
	missing_stats = missing_stats.rename_axis('Column Name').reset_index() 
	missing_stats['Null Percentage'] = (missing_stats['Null Count'] / len(data)) * 100    

	return missing_stats

def corr_plot(data):
	'''
	Get correlation matrix of numeric columns in the data and make a correlation plot 
    
    Parameter description:
		(1) data - Dataframe for which correlation needs to be found out
	'''
    
	corr = data.corr()
    
	corr_unpivot = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
	corr_unpivot.columns = ['x', 'y', 'value']
	n_colors = 256 # Use 256 colors for the diverging color palette
	palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette    
    
	heatmap(x=corr_unpivot['x'], y=corr_unpivot['y'], size=corr_unpivot['value'].abs(), color=corr_unpivot['value'], color_range=(-1,1), palette=palette)
		
	return corr

def bar_plot(data, x_var, y_var, hue_var):
	''' 
	Bar plot visualization between a continuous and categorical variable

    Parameter description:
		(1) data    - Dataframe for which vatiables plot is to be made
		(2) x_var   - Variable for horizontal axis. Should be categorical
		(3) y_var   - Variable for vertical axis. Should be continuous
		(4) hue_var - Variable used to sub-divide bars. Set hue = None if not applicable
	'''
	
	sns.barplot(x=x_var, y=y_var, hue=hue_var, data=data,
			palette='hls',
			capsize=0.05,             
			saturation=8,             
			errcolor='gray', 
			errwidth=2,  
			ci='sd'   
			)
	plt.show()

def box_plot(data, x_var, y_var):
	''' 
	Box plot visualization between a continuous and categorical variable	

    Parameter description:
		(1) data    - Dataframe for which vatiables plot is to be made
		(2) x_var   - Variable for horizontal axis. Should be categorical
		(3) y_var   - Variable for vertical axis. Should be continuous
	'''
	
	sns.boxplot(x=x_var, y=y_var, data=data,
				width=0.3);
	plt.show()
	
def scatter_plot(data, x_var, y_var, hue_var):
	'''
	Scatter plot visualization between two continuous variables
	
    Parameter description:
		(1) data    - Dataframe for which vatiables plot is to be made
		(2) x_var   - Variable for horizontal axis. Should be continuous
		(3) y_var   - Variable for vertical axis. Should be continuous
		(4) hue_var - Variable used to cluster scatters. Set hue = None if not applicable
	'''

	# Set style of scatterplot
	sns.set_context("notebook", font_scale=1.1)
	sns.set_style("ticks")

	# Create scatterplot of dataframe
	sns.lmplot(x=x_var, y=y_var, data=data, 
			   fit_reg=False, 			   # Don't fix a regression line
			   hue=hue_var,			 	   # Set color
			   scatter_kws={"marker": "D", # Set marker style
							"s": 100}) 	   # S marker size

	plt.xlabel(x_var) 			 		   # Set x-axis label
	plt.ylabel(y_var) 		 			   # Set y-axis label

def pairwise_plot(data, hue_var):
	''' 
	Pairwise plot visualization between multiple variables	

    Parameter description:
		(1) data    - Dataframe for which vatiables plot is to be made
		(2) hue_var - Changes the histograms to KDE plots to facilitate comparisons between multiple distributions. Set to None if not applicable
	'''
	
	sns.pairplot(data=data, hue=hue_var);

def histogram(x_var):
	''' 
	Histogram for a numeric variable	

    Parameter description:
		(1) x_var - Series of a numeric variable for which to plot histogram
	'''
	
	sns.distplot(x_var)
	
# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 3 - EXECUTE EDA
# -------------------------------------------------------------------------------------------------------------------------------------

if data_types == True:
	print('Get data types ...')
	data_type = pd.DataFrame(data_train.dtypes).reset_index()     
	data_type.columns = ['Column Name', 'Data Type']  

if missing_data_stats == True:
	print('Generating missing data stats ...')
	missing_stats = miss_data_stats(data_train)
    
if descriptive_stats == True:
	print('Generating descriptive statistics ...')
	desc_stats = data_train.describe()

if correlation_plot == True:
	print('Plotting correlation matrix ...')
	corr = corr_plot(data_train)

# -------------------------------------------------------------------------------------------------------------------------------------
# 					STEP 4 - SAVE EDA ANALYSIS
# -------------------------------------------------------------------------------------------------------------------------------------

print('Saving analysis ...')
data_type.to_csv(root_directory + '\\EDA\\train-data_types.csv', index=False)
missing_stats.to_csv(root_directory + '\\EDA\\train-missing_data_stats.csv', index=False)
desc_stats.to_csv(root_directory + '\\EDA\\train-eda.csv')
corr.to_csv(root_directory + '\\EDA\\train-corr.csv')

# -------------------------------------------------------------------------------------------------------------------------------------
# 						STEP 5 - AD-HOC EDA
# -------------------------------------------------------------------------------------------------------------------------------------

# Plot pairwise plots
pairwise_plot(data=data_train, hue_var='species')

# Plot histogram of a continuous variable
histogram(x_var=data_train['survived'])

# Plot bar chart between a categorical (x) and a continuous (y) variable
bar_plot(data=data_train, x_var='sex', y_var='survived', hue_var='class')

# Plot box plots between a categorical (x) and a continuous (y) variable
box_plot(data=data_train, x_var='species', y_var='sepal_length')

# Plot scatter plots between two continuous variables
scatter_plot(data=data_train, x_var='x', y_var='y', hue_var='z')























