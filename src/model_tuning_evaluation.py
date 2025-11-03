import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import joblib

# Load preprocessed and encoded dataset
data = pd.read_csv(r'C:\Users\Aiza anjum\cern-ml-project\data\cms_open_data_features.csv')

# One-hot encode categorical features
data_encoded = pd.get_dummies(data, columns=['Type1', 'Type2'])

# Define features and target
X = data_encoded.drop(columns=['M'])
y = data_encoded['M']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define model
rf = RandomForestRegressor(random_state=42)

# Define parameter distribution with valid max_features options
param_dist = {
    'n_estimators': randint(50, 100),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2']
}

# Setup Randomized Search with 3-fold CV
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=2,
    error_score='raise'
)

# Run hyperparameter tuning
random_search.fit(X_train, y_train)

print("Best parameters found:")
print(random_search.best_params_)

# Evaluate the best estimator on test set
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Test set performance:")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# 3-fold cross-validation on whole data with best params
cv_scores = cross_val_score(best_rf, X, y, cv=3, scoring='r2')
print(f"3-fold CV R2 scores: {cv_scores}")
print(f"Mean CV R2 score: {cv_scores.mean():.4f}")

joblib.dump(best_rf, r'C:\Users\Aiza anjum\cern-ml-project\models\best_random_forest.pkl')

print("Best model saved as 'best_random_forest.pkl'")
