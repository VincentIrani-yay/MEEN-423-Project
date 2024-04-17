
import pandas as pd

# Load data from the CSV file
data = pd.read_csv('Concrete_Data.csv')

# Define the feature columns and target variable
feature_columns = ['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)', 'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)', 'Superplasticizer (component 5)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)']
target_column = 'Concrete Compressive Strength MPA'

# Select features and target from the dataset
X = data[feature_columns]  # Features (input)
y = data[target_column]    # Target variable (output)
