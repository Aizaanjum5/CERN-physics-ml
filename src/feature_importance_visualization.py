import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.model_tuning_evaluation import best_rf

# Load and encode data as in previous steps
data = pd.read_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data_features.csv')
data_encoded = pd.get_dummies(data, columns=['Type1', 'Type2'])
X = data_encoded.drop(columns=['M'])
feature_names = X.columns

# ... Run RandomizedSearchCV and get best_rf as before ...

# After training and evaluation, get feature importances:
importances = best_rf.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Print top features
print("Top 10 important features:")
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot feature importances
plt.figure(figsize=(10,6))
plt.title("Top 10 Feature Importances (Random Forest)")
plt.bar(range(10), importances[indices[:10]], align="center")
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45)
plt.tight_layout()
plt.show()