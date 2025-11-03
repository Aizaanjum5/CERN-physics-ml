import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the feature-engineered data
data = pd.read_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data_features.csv')

# Encode categorical columns 'Type1' and 'Type2' using one-hot encoding
data_encoded = pd.get_dummies(data, columns=['Type1', 'Type2'])

# Define features and target variable
feature_cols = [col for col in data_encoded.columns if col != 'M']
X = data_encoded[feature_cols]
y = data_encoded['M']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
print("Model performance:")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
