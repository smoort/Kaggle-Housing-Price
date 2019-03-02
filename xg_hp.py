# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
print("Modules imported")

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = 'data/train.csv'
home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

# Create X
X = home_data.drop(['Id', 'SalePrice'],axis=1)

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

train_X['label'] = 1
val_X['label'] = 0

# Concat
concat_df = pd.concat([train_X, val_X])
print('after concat')

# Create your dummies
features_df = pd.get_dummies(concat_df, dummy_na=True)
#features_df = concat_df.fillna(method='ffill')
print('after dummies')

# Fill holes by imputation
imputed_features_df = features_df.copy()
my_imputer = SimpleImputer()
imputed_features_nd = my_imputer.fit_transform(imputed_features_df)
imputed_features_df = pd.DataFrame(imputed_features_nd, columns=features_df.columns)

# Split your data
train_df = imputed_features_df[imputed_features_df['label'] == 1]
score_df = imputed_features_df[imputed_features_df['label'] == 0]
print('after split')
print("Data preprocessed")

# Drop your labels
train_X = train_df.drop('label', axis=1)
val_X = score_df.drop('label', axis=1)
print('after drop')

# Define the model. Set random_state to 1
rf_model = XGBRegressor()
rf_model.fit(train_X, train_y, verbose=False)
print("Training1 completed")

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model21: {:,.0f}".format(rf_val_mae))

# Define the model.
rf_model = XGBRegressor(n_estimators=1000)
rf_model.fit(train_X, train_y, early_stopping_rounds=50, 
             eval_set=[(val_X, val_y)], verbose=False)
print("Training2 completed")

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model22: {:,.0f}".format(rf_val_mae))

# Define the model.
rf_model = XGBRegressor(n_estimators=5000, learning_rate=0.05)
rf_model.fit(train_X, train_y, early_stopping_rounds=100, 
             eval_set=[(train_X, train_y)], verbose=False)
print("Training3 completed")

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model23: {:,.0f}".format(rf_val_mae))