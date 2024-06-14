import os
import pyodbc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib
from openpyxl import Workbook

# Connection parameters (use environment variables for sensitive data)
server = os.getenv('DB_SERVER', 'localhost')
database = os.getenv('DB_NAME', 'CryptoPredictor')
username = os.getenv('DB_USER', 'sa')
password = os.getenv('DB_PASSWORD', 'Pakistan@12')


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
def load_data(features, target_variable):
    conn = create_connection()
    if conn is not None:
        try:
            sql_query = 'SELECT {} FROM Data'.format(
                ', '.join([f'[{feature}]' for feature in features + [target_variable] + ['Date']]))
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()


# Function to preprocess data
def preprocess_data(data, features, scaler_path):
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).astype('int64') / 10 ** 9

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=features, inplace=True)

    numeric_features = data[features].select_dtypes(include=[np.number])

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(numeric_features)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)  # Add channel dimension

    return X_scaled, data


# Function to make predictions
def make_predictions(target_variable, sheet_name, model_path, scaler_path):
    feature_file = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\Feature_Selection_Results.xlsx'
    features = pd.read_excel(feature_file, sheet_name=sheet_name).iloc[:, 0].tolist()
    data = load_data(features, target_variable)
    if not data.empty:
        X_scaled, preprocessed_data = preprocess_data(data, features, scaler_path)
        model = tf.keras.models.load_model(model_path)

        last_date_data = X_scaled[-1].reshape(1, -1, 1)
        prediction = model.predict(last_date_data)

        prediction_date = pd.to_datetime(preprocessed_data['Date'].values[-1], unit='s') + pd.Timedelta(days=1)
        result = {
            'Sheet Name': sheet_name,
            'Target Variable': target_variable,
            'Data Date': pd.to_datetime(preprocessed_data['Date'].values[-1], unit='s'),
            'Predicted Value': prediction[0][0],
            'Prediction Date': prediction_date
        }
        return result
    else:
        print(f"No data loaded to process for {target_variable} with {sheet_name}.")
        return None


# Define paths and variables
output_folder = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\CNN'
output_file = os.path.join(output_folder, 'predictions_CNN_Model.xlsx')
target_features_map = {
    'target_BTC_Price': ['target_BTC_Pric_Correlation', 'target_BTC_Pric_Granger', 'target_BTC_Pric_RandomForest',
                         'target_BTC_Pric_SHAP'],
    'Target_XRP_Price': ['Target_XRP_Pric_Correlation', 'Target_XRP_Pric_Granger', 'Target_XRP_Pric_RandomForest',
                         'Target_XRP_Pric_SHAP'],
    'Target_SOL_Price': ['Target_SOL_Pric_Correlation', 'Target_SOL_Pric_Granger', 'Target_SOL_Pric_RandomForest',
                         'Target_SOL_Pric_SHAP'],
    'Target_ETH_Price': ['Target_ETH_Pric_Correlation', 'Target_ETH_Pric_Granger', 'Target_ETH_Pric_RandomForest',
                         'Target_ETH_Pric_SHAP'],
    'Target_BNB_Price': ['Target_BNB_Pric_Correlation', 'Target_BNB_Pric_Granger', 'Target_BNB_Pric_RandomForest',
                         'Target_BNB_Pric_SHAP']
}

# Perform predictions
results = []
for target_variable, feature_sheets in target_features_map.items():
    for sheet_name in feature_sheets:
        model_path = os.path.join(output_folder, f'{target_variable}_CNN_Model_{sheet_name}.h5')
        scaler_path = os.path.join(output_folder, f'{target_variable}_RobustScaler_{sheet_name}.pkl')
        result = make_predictions(target_variable, sheet_name, model_path, scaler_path)
        if result:
            results.append(result)

# Save results to Excel
if results:
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_file, index=False)
    print(f"Predictions saved to {output_file}")
else:
    print("No predictions were made.")
