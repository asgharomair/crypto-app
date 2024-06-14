import pyodbc
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# Database credentials
server = 'localhost'
database = 'CryptoPredictor'
username = 'sa'
password = 'Pakistan@12'


# Function to create database connection
def create_connection():
    try:
        conn = pyodbc.connect(
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )
        return conn
    except pyodbc.Error as e:
        print("Error in connection: ", e)
        return None


# Function to load data from the database
def load_data(features):
    conn = create_connection()
    if conn is not None:
        try:
            sql_query = 'SELECT {} FROM Data'.format(', '.join([f'[{feature}]' for feature in features]))
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()


# Function to preprocess data
def preprocess_data(data, features, scaler):
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).astype('int64') / 10 ** 9

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=features, inplace=True)

    numeric_features = data[features].select_dtypes(include=[np.number])

    X_scaled = scaler.transform(numeric_features)

    # Reshape data for GRU input
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

    return X_scaled, data['Date']


# List of models and scalers
models_and_scalers = [
    ('Target_BNB_Price', 'Target_BNB_Pric_Correlation'),
    ('Target_BNB_Price', 'Target_BNB_Pric_Granger'),
    ('Target_BNB_Price', 'Target_BNB_Pric_RandomForest'),
    ('Target_BNB_Price', 'Target_BNB_Pric_SHAP'),
    ('target_BTC_Price', 'target_BTC_Pric_Correlation'),
    ('target_BTC_Price', 'target_BTC_Pric_Granger'),
    ('target_BTC_Price', 'target_BTC_Pric_RandomForest'),
    ('target_BTC_Price', 'target_BTC_Pric_SHAP'),
    ('Target_ETH_Price', 'Target_ETH_Pric_Correlation'),
    ('Target_ETH_Price', 'Target_ETH_Pric_Granger'),
    ('Target_ETH_Price', 'Target_ETH_Pric_RandomForest'),
    ('Target_ETH_Price', 'Target_ETH_Pric_SHAP'),
    ('Target_SOL_Price', 'Target_SOL_Pric_Correlation'),
    ('Target_SOL_Price', 'Target_SOL_Pric_Granger'),
    ('Target_SOL_Price', 'Target_SOL_Pric_RandomForest'),
    ('Target_SOL_Price', 'Target_SOL_Pric_SHAP'),
    ('Target_XRP_Price', 'Target_XRP_Pric_Correlation'),
    ('Target_XRP_Price', 'Target_XRP_Pric_Granger'),
    ('Target_XRP_Price', 'Target_XRP_Pric_RandomForest'),
    ('Target_XRP_Price', 'Target_XRP_Pric_SHAP'),
]

# File paths
model_dir = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\GRU Model'
output_file = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\predictions_GRU_Model.xlsx'
feature_file = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\Feature_Selection_Results.xlsx'

# Initialize results DataFrame
results_df = pd.DataFrame(
    columns=['Sheet Name', 'Target Variable', 'Feature Sheet', 'Data Date', 'Predicted Value', 'Prediction Date'])

# Iterate over each model and make predictions
for target_variable, sheet_name in models_and_scalers:
    try:
        # Load the model and scaler
        model_path = os.path.join(model_dir, f'{target_variable}_GRU_Model_{sheet_name}.h5')
        scaler_path = os.path.join(model_dir, f'{target_variable}_RobustScaler_{sheet_name}.pkl')
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Load feature names
        features = pd.read_excel(feature_file, sheet_name=sheet_name).iloc[:, 0].tolist()

        # Load data
        data = load_data(features + ['Date'])

        # Preprocess data
        X_scaled, data_dates = preprocess_data(data, features, scaler)

        if X_scaled.shape[0] == 0:
            print(f"No data available for {target_variable} using {sheet_name}.")
            continue

        # Predict the last value
        last_data_date = data_dates.iloc[-1]
        prediction = model.predict(X_scaled[-1].reshape(1, -1, 1))

        # Save the results
        result = {
            'Sheet Name': sheet_name,
            'Target Variable': target_variable,
            'Feature Sheet': sheet_name,
            'Data Date': pd.to_datetime(last_data_date * 10 ** 9),
            'Predicted Value': prediction[0, 0],
            'Prediction Date': pd.to_datetime(last_data_date * 10 ** 9) + pd.Timedelta(days=1)
        }
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
    except Exception as e:
        print(f"Error processing {target_variable} with {sheet_name}: {e}")

# Save results to Excel
results_df.to_excel(output_file, index=False)
print(f"Predictions saved to {output_file}")
