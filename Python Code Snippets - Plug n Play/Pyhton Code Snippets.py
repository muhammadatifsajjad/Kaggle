# Get distinct values in a column
unique_stores = data['tes']['air_store_id'].unique()

# For each distinct value in unique_stores, add day of week from 1 to 7
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

# Convert factors to numeric after concatenating train and test data sets
lbl = preprocessing.LabelEncoder()
a = lbl.fit(pd.DataFrame(pd.concat([train['air_store_id'], test['air_store_id']])))
train['air_store_id2'] = a.transform(train['air_store_id'])

# Get all columns of data frame excluding specified columns
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
