import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset using the correct relative path
df_q1 = pd.read_csv("../data/Question 1 datasets _e04481f3a334d40a121407d1eb29e12b.csv")

# Drop the 'Index' column and separate features (X) and target (y)
df_q1 = df_q1.drop('Index', axis=1)
X = df_q1.drop('Days to Failure', axis=1)
y = df_q1['Days to Failure']

# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Initialize and fit the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully.")

# Predict on test data
y_pred = rf_model.predict(X_test)

# Calculate and print RMSE and R-squared on test data
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
r2_test = r2_score(y_test, y_pred)
print("\nModel Performance on Test Data:")
print(f"RMSE: {rmse_test:.2f}")
print(f"R-squared: {r2_test:.2f}")
print("-" * 30)

# Perform 5-fold cross-validation
rmse_cv_scores = -cross_val_score(rf_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
r2_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
print("Cross-Validation Results:")
print(f"Average RMSE from 5-fold cross-validation: {rmse_cv_scores.mean():.2f}")
print(f"Average R-squared from 5-fold cross-validation: {r2_cv_scores.mean():.2f}")
print("-" * 30)

# Save predictions to the output folder
try:
    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    predictions_df.to_csv('../output/rf_predictions.csv', index=False)
    print("rf_predictions.csv successfully saved to the 'output' folder.")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")