import pandas as pd

# Load CMS open data

data = pd.read_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data.csv')


# Show a preview and summary stats
print("First 5 rows:")
print(data.head())
print("\nData info:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())

# drop rows with missing values
data_clean = data.dropna()

# Example feature engineering: sum of energies
if 'E1' in data_clean.columns and 'E2' in data_clean.columns:
    data_clean['E_sum'] = data_clean['E1'] + data_clean['E2']

# Save cleaned and feature-enriched version
data_clean.to_csv('../data/cms_events_clean.csv', index=False)

print("\nSaved cleaned data with new features to: ../data/cms_events_clean.csv")
