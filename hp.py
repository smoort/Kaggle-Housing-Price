# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print("Modules imported")

# Read the file with training data
iowa_file_path = 'data/train.csv'
rf_model_on_full_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = rf_model_on_full_data.SalePrice

# Create training data X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = rf_model_on_full_data[features]

# fit rf_model_on_full_data on all data from the 

# Specify Model
iowa_model = RandomForestRegressor(random_state=1)
# Fit Model
iowa_model.fit(X, y)

print("Training completed")

# path to file you will use for predictions
test_data_path = 'data/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# make predictions which we will submit. 
test_preds = iowa_model.predict(test_X)

print("Validation Completed")

print("Adios Amigo")