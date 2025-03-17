import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
file_path = "mangrove_metrics_summary.csv"  # Change to your CSV file path
df = pd.read_csv(file_path)

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Set the date as index (for time series modeling)
df.set_index('date', inplace=True)

# Selecting target variables
target_cols = ['total_carbon_stock_tonnes', 'total_CO2_absorption_tonnes_per_year']

# Creating lag features for each target variable
for target in target_cols:
    df[f'{target}_lag1'] = df[target].shift(1)
    df[f'{target}_lag2'] = df[target].shift(2)

# Drop rows with NaN values (due to shifting)
df.dropna(inplace=True)

# Selecting feature columns (lag values)
feature_cols = [f'{col}_lag1' for col in target_cols] + [f'{col}_lag2' for col in target_cols]

# Define X (features) and y (targets)
X = df[feature_cols]
y = df[target_cols]

# Splitting data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ------------------ Random Forest Model ------------------

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save Random Forest Model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Predicting on test data
rf_preds = rf_model.predict(X_test)
rf_preds_df = pd.DataFrame(rf_preds, index=y_test.index, columns=target_cols)

# ------------------ SARIMA Model ------------------

sarima_preds = {}
sarima_models = {}

for target in target_cols:
    # Fit SARIMA model
    sarima_model = SARIMAX(df[target], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit(disp=False)

    # Save SARIMA model
    with open(f'sarima_model_{target}.pkl', 'wb') as f:
        pickle.dump(sarima_result, f)

    # Forecast on the test set
    sarima_preds[target] = sarima_result.predict(start=y_test.index[0], end=y_test.index[-1])
    sarima_models[target] = sarima_result

# Convert SARIMA predictions to DataFrame
sarima_preds_df = pd.DataFrame(sarima_preds, index=y_test.index)

# ------------------ Performance Metrics ------------------

def evaluate_model(y_true, y_pred, model_name):
    print(f"\nPerformance Metrics for {model_name}:")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

# Evaluate models
evaluate_model(y_test, rf_preds_df, "Random Forest")
evaluate_model(y_test, sarima_preds_df, "SARIMA")

# ------------------ Plot Comparison Graphs ------------------

plt.figure(figsize=(14, 6))

for i, target in enumerate(target_cols):
    plt.subplot(1, 2, i + 1)
    plt.plot(y_test.index, y_test[target], label='Actual', marker='o', linestyle='dashed')
    plt.plot(y_test.index, rf_preds_df[target], label='Random Forest', linestyle='solid')
    plt.plot(y_test.index, sarima_preds_df[target], label='SARIMA', linestyle='dotted')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.title(f'Comparison for {target}')
    plt.legend()
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("Models saved successfully!")