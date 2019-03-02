# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
print("Modules imported")

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'data/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice

'''
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
print("Data preprocessed")


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
print("Training1 completed")

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model1: {:,.0f}".format(rf_val_mae))


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(n_estimators=85, random_state=1)
rf_model.fit(train_X, train_y)
print("Training2 completed")

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model2: {:,.0f}".format(rf_val_mae))
'''

# Create new X

new_X = home_data.drop(['Id', 'SalePrice'],axis=1)
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'MSZoning', 'Street', 'Neighborhood']
#new_X = home_data[features]
#new_X = new_X.fillna(method='ffill')
#print(new_X.head())
#new_X = pd.get_dummies(new_X, dummy_na=True)
#print(new_X.head())

#print(new_X.describe)

# Split into validation and training data
new_train_X, new_val_X, new_train_y, new_val_y = train_test_split(new_X, y, random_state=1)
print("Data preprocessed")


new_train_X['label'] = 1
new_val_X['label'] = 0
#print(new_train_X.describe)
#print(new_val_X.describe)

# Concat
concat_df = pd.concat([new_train_X, new_val_X])
print('after concat')

# Create your dummies
features_df = pd.get_dummies(concat_df, dummy_na=True)
#features_df = concat_df.fillna(method='ffill')
imputed_features_df = features_df.copy()
print(features_df.columns)

my_imputer = SimpleImputer()
imputed_features_nd = my_imputer.fit_transform(imputed_features_df)
imputed_features_df = pd.DataFrame(imputed_features_nd, columns=features_df.columns)
print('after dummies')
#imputed_features_df.colums = features_df.columns
print(imputed_features_df.columns)
print(imputed_features_df.describe)
print(imputed_features_df.label)

# Split your data
train_df = imputed_features_df[imputed_features_df['label'] == 1]
score_df = imputed_features_df[imputed_features_df['label'] == 0]
print('after split')


# Drop your labels
new_train_X = train_df.drop('label', axis=1)
new_val_X = score_df.drop('label', axis=1)
print('after drop')

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(new_train_X, new_train_y)
print("Training1 completed")

rf_val_predictions = rf_model.predict(new_val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, new_val_y)

print("Validation MAE for Random Forest Model21: {:,.0f}".format(rf_val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(n_estimators=20, random_state=1)
rf_model.fit(new_train_X, new_train_y)
print("Training2 completed")

rf_val_predictions = rf_model.predict(new_val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, new_val_y)

print("Validation MAE for Random Forest Model22: {:,.0f}".format(rf_val_mae))