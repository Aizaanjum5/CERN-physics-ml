import pandas as pd

# Load the cleaned data
data = pd.read_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data_clean.csv')

# sum of energies if columns exist
if 'E1' in data.columns and 'E2' in data.columns:
    data['E_sum'] = data['E1'] + data['E2']


# difference in transverse momentum
if 'pt1' in data.columns and 'pt2' in data.columns:
    data['pt_diff'] = abs(data['pt1'] - data['pt2'])

# Save the feature-enhanced dataset
data.to_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data_features.csv', index=False)

print("Feature engineering complete and saved as cms_open_data_features.csv")
