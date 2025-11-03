import pandas as pd

# Load data
data = pd.read_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data.csv')

# Preview data
print("First 5 rows:")
print(data.head())

print("\nData info:")
print(data.info())

print("\nSummary statistics:")
print(data.describe())

# Find missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Drop rows with missing values
data_clean = data.dropna()

# sum of energies (example)
data_clean['E_sum'] = data_clean['E1'] + data_clean['E2']

# Save cleaned version
data_clean.to_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data_clean.csv', index=False)

print("\nCleaned data saved as 'cms_open_data_clean.csv'")
