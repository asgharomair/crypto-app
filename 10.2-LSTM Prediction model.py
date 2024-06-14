import pyodbc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib
import os

# Connection parameters
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
                ', '.join([f'[{feature}]' for feature in features + [target_variable, 'Date']])
            )
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# Function to preprocess data
def preprocess_data(row, features, scaler_path):
    row = row[features].values.reshape(1, -1)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(row)
    # Reshape data for LSTM input
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))
    return X_scaled

# Function to find the last row with complete data
def find_last_complete_row(data, features):
    for i in range(len(data) - 1, -1, -1):
        if not data.iloc[i][features].isnull().any():
            return data.iloc[i]
    return None

# Function to predict using the model
def predict_with_model(sheet_name, model_path, scaler_path, features, target_variable):
    data = load_data(features, target_variable)
    if not data.empty:
        last_complete_row = find_last_complete_row(data, features)
        if last_complete_row is not None:
            model = tf.keras.models.load_model(model_path)
            date = last_complete_row['Date']
            X_scaled = preprocess_data(last_complete_row, features, scaler_path)
            prediction = model.predict(X_scaled)[0][0]
            return {
                'Sheet Name': sheet_name,
                'Target Variable': target_variable,
                'Data Date': pd.to_datetime(date, unit='s'),
                'Predicted Value': prediction,
                'Prediction Date': pd.to_datetime(date, unit='s') + pd.Timedelta(days=1)
            }
        else:
            print("No complete data row found")
    else:
        print("No data loaded")
    return None

# Paths to model and scaler
output_folder = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\LSTM Model'
model_files = [f for f in os.listdir(output_folder) if f.endswith('.h5')]

# Prepare predictions DataFrame
predictions = []

# Iterate through each model and make a single prediction
for model_file in model_files:
    sheet_name = model_file.split('_LSTM_Model_')[1].replace('.h5', '')
    target_variable = model_file.split('_LSTM_Model_')[0]
    scaler_file = target_variable + '_RobustScaler_' + sheet_name + '.pkl'
    scaler_path = os.path.join(output_folder, scaler_file)
    model_path = os.path.join(output_folder, model_file)
    feature_file = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\Feature_Selection_Results.xlsx'
    features = pd.read_excel(feature_file, sheet_name=sheet_name).iloc[:, 0].tolist()
    prediction = predict_with_model(sheet_name, model_path, scaler_path, features, target_variable)
    if prediction:
        predictions.append(prediction)

# Convert predictions list to DataFrame
predictions_df = pd.DataFrame(predictions)

# Save predictions to an Excel file
output_file = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\predictions_LSTM_Model.xlsx'
predictions_df.to_excel(output_file, index=False)
print(f"Predictions saved to {output_file}")
