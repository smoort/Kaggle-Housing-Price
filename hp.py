# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
print("Modules imported")

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'data/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
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

# Create new X

#new_X = home_data.drop(['Id', 'SalePrice'],axis=1)
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


# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(new_train_X, new_train_y)
print("Training1 completed")

rf_val_predictions = rf_model.predict(new_val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, new_val_y)

print("Validation MAE for Random Forest Model21: {:,.0f}".format(rf_val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(n_estimators=2000, random_state=1)
rf_model.fit(new_train_X, new_train_y)
print("Training1 completed")

rf_val_predictions = rf_model.predict(new_val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, new_val_y)

print("Validation MAE for Random Forest Model22: {:,.0f}".format(rf_val_mae))

'''


# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print("Modules imported")

# Read the file with training data
iowa_file_path = 'data/train.csv'
rf_model_on_full_data = pd.read_csv(iowa_file_path)
print(rf_model_on_full_data.head())

# Create target object and call it y
y = rf_model_on_full_data.SalePrice

# Create training data X
#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
#X = rf_model_on_full_data[features]
X = rf_model_on_full_data.drop(['Id','SalePrice'], axis=1)
print(X.head())

# fit rf_model_on_full_data on all data from the 

# Specify Model
iowa_model = RandomForestRegressor(random_state=1)
# Fit Model
#iowa_model.fit(X, y)

print("Training completed")

# path to file you will use for predictions
test_data_path = 'data/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
#test_X = test_data[features]
test_X = test_data.drop(['Id'], axis=1)

# make predictions which we will submit. 
test_preds = iowa_model.predict(test_X)

print("Validation Completed")

print("Adios Amigo")

'''