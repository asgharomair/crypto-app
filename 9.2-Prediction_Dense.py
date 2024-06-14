import pyodbc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import joblib
import os

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

# Function to get available columns from the database
def get_available_columns():
    conn = create_connection()
    if conn is not None:
        try:
            query = "SELECT * FROM Data WHERE 1=0"  # Select no rows, just get the schema
            df = pd.read_sql(query, conn)
            conn.close()
            return df.columns.tolist()
        except Exception as e:
            print("Error getting columns: ", e)
            return []
    else:
        return []

# Function to load data from the database
def load_data(features, target_variable):
    conn = create_connection()
    if conn is not None:
        try:
            available_columns = get_available_columns()
            selected_columns = [col for col in features + [target_variable, 'Date'] if col in available_columns]
            sql_query = 'SELECT {} FROM Data'.format(
                ', '.join([f'[{col}]' for col in selected_columns])
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
def preprocess_data(data, features, scaler_path):
    data['Date'] = pd.to_datetime(data['Date'])
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=features, inplace=True)

    numeric_features = data[features].select_dtypes(include=[np.number])
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(numeric_features)

    return X_scaled, data

# Function to load feature sets from Excel
def load_feature_sets():
    feature_file = r'C:\omair\Projects\CapStone Project\Crypto Project results\Feature_Selection_Results.xlsx'
    sheets = ['target_BTC_Pric_Correlation', 'target_BTC_Pric_Granger', 'target_BTC_Pric_RandomForest',
              'target_BTC_Pric_SHAP', 'Target_XRP_Pric_Correlation', 'Target_XRP_Pric_Granger',
              'Target_XRP_Pric_RandomForest', 'Target_XRP_Pric_SHAP', 'Target_SOL_Pric_Correlation',
              'Target_SOL_Pric_Granger', 'Target_SOL_Pric_RandomForest', 'Target_SOL_Pric_SHAP',
              'Target_ETH_Pric_Correlation', 'Target_ETH_Pric_Granger', 'Target_ETH_Pric_RandomForest',
              'Target_ETH_Pric_SHAP', 'Target_BNB_Pric_Correlation', 'Target_BNB_Pric_Granger',
              'Target_BNB_Pric_RandomForest', 'Target_BNB_Pric_SHAP']
    feature_sets = {}

    for sheet in sheets:
        features = pd.read_excel(feature_file, sheet_name=sheet).iloc[:, 0].tolist()
        feature_sets[sheet] = features

    return feature_sets

# Function to predict using the model
def load_data(features, target_variable):
    conn = create_connection()
    if conn is not None:
        try:
            available_columns = get_available_columns()
            print(f"Available columns: {available_columns}")
            selected_columns = [col for col in features + [target_variable, 'Date'] if col in available_columns]
            print(f"Selected columns for {target_variable}: {selected_columns}")
            sql_query = 'SELECT {} FROM Data'.format(
                ', '.join([f'[{col}]' for col in selected_columns])
            )
            print(f"SQL Query: {sql_query}")
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def predict_with_model(sheet_name, model_path, scaler_path, log):
    feature_sets = load_feature_sets()
    features = feature_sets[sheet_name]
    target_variable = sheet_name.split('_')[1]
    data = load_data(features, target_variable)

    if data.empty:
        log.append({"Sheet Name": sheet_name, "Reason": "No data available"})
        print(f"No data available for {sheet_name}")
        return None

    # Print the columns of the DataFrame for debugging
    print(f"Data columns for {sheet_name}:", data.columns)

    # Find the last row with complete data
    last_row = None
    for i in range(len(data) - 1, -1, -1):
        if not data.iloc[i].isnull().any():
            last_row = data.iloc[i]
            break
    if last_row is None:
        log.append({"Sheet Name": sheet_name, "Reason": "No complete data row found"})
        print(f"No complete data row found for {sheet_name}")
        return None

    # Print the last row for debugging
    print(f"Last row data for {sheet_name}:", last_row)

    # Preprocess data
    try:
        X_scaled, data = preprocess_data(data, features, scaler_path)
    except Exception as e:
        log.append({"Sheet Name": sheet_name, "Reason": f"Preprocessing error: {str(e)}"})
        print(f"Preprocessing error for {sheet_name}: {str(e)}")
        return None

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        log.append({"Sheet Name": sheet_name, "Reason": f"Model loading error: {str(e)}"})
        print(f"Model loading error for {sheet_name}: {str(e)}")
        return None

    # Predict
    try:
        pred_input = X_scaled[-1].reshape(1, -1)
        prediction = model.predict(pred_input)[0][0]
    except Exception as e:
        log.append({"Sheet Name": sheet_name, "Reason": f"Prediction error: {str(e)}"})
        print(f"Prediction error for {sheet_name}: {str(e)}")
        return None

    # Display results
    prediction_date = last_row['Date'] + pd.Timedelta(days=1)
    print(f"Prediction for {sheet_name}:")
    print(f"Data Date: {last_row['Date']}")
    print(f"Predicted Value: {prediction}")
    print(f"Prediction Date: {prediction_date}")
    print()

    return {
        "Sheet Name": sheet_name,
        "Target Variable": target_variable,
        "Feature Sheet": sheet_name,
        "Data Date": last_row['Date'],
        "Predicted Value": prediction,
        "Prediction Date": prediction_date
    }

# Paths to models and scalers
output_folder = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\BTC Model'

model_paths = {
    'target_BTC_Pric_Correlation': os.path.join(output_folder, 'target_BTC_Price_Dense_Model_target_BTC_Pric_Correlation.h5'),
    'target_BTC_Pric_Granger': os.path.join(output_folder, 'target_BTC_Price_Dense_Model_target_BTC_Pric_Granger.h5'),
    'target_BTC_Pric_RandomForest': os.path.join(output_folder, 'target_BTC_Price_Dense_Model_target_BTC_Pric_RandomForest.h5'),
    'target_BTC_Pric_SHAP': os.path.join(output_folder, 'target_BTC_Price_Dense_Model_target_BTC_Pric_SHAP.h5'),
    'Target_XRP_Pric_Correlation': os.path.join(output_folder, 'Target_XRP_Price_Dense_Model_Target_XRP_Pric_Correlation.h5'),
    'Target_XRP_Pric_Granger': os.path.join(output_folder, 'Target_XRP_Price_Dense_Model_Target_XRP_Pric_Granger.h5'),
    'Target_XRP_Pric_RandomForest': os.path.join(output_folder, 'Target_XRP_Price_Dense_Model_Target_XRP_Pric_RandomForest.h5'),
    'Target_XRP_Pric_SHAP': os.path.join(output_folder, 'Target_XRP_Price_Dense_Model_Target_XRP_Pric_SHAP.h5'),
    'Target_SOL_Pric_Correlation': os.path.join(output_folder, 'Target_SOL_Price_Dense_Model_Target_SOL_Pric_Correlation.h5'),
    'Target_SOL_Pric_Granger': os.path.join(output_folder, 'Target_SOL_Price_Dense_Model_Target_SOL_Pric_Granger.h5'),
    'Target_SOL_Pric_RandomForest': os.path.join(output_folder, 'Target_SOL_Price_Dense_Model_Target_SOL_Pric_RandomForest.h5'),
    'Target_SOL_Pric_SHAP': os.path.join(output_folder, 'Target_SOL_Price_Dense_Model_Target_SOL_Pric_SHAP.h5'),
    'Target_ETH_Pric_Correlation': os.path.join(output_folder, 'Target_ETH_Price_Dense_Model_Target_ETH_Pric_Correlation.h5'),
    'Target_ETH_Pric_Granger': os.path.join(output_folder, 'Target_ETH_Price_Dense_Model_Target_ETH_Pric_Granger.h5'),
    'Target_ETH_Pric_RandomForest': os.path.join(output_folder, 'Target_ETH_Price_Dense_Model_Target_ETH_Pric_RandomForest.h5'),
    'Target_ETH_Pric_SHAP': os.path.join(output_folder, 'Target_ETH_Price_Dense_Model_Target_ETH_Pric_SHAP.h5'),
    'Target_BNB_Pric_Correlation': os.path.join(output_folder, 'Target_BNB_Price_Dense_Model_Target_BNB_Pric_Correlation.h5'),
    'Target_BNB_Pric_Granger': os.path.join(output_folder, 'Target_BNB_Price_Dense_Model_Target_BNB_Pric_Granger.h5'),
    'Target_BNB_Pric_RandomForest': os.path.join(output_folder, 'Target_BNB_Price_Dense_Model_Target_BNB_Pric_RandomForest.h5'),
    'Target_BNB_Pric_SHAP': os.path.join(output_folder, 'Target_BNB_Price_Dense_Model_Target_BNB_Pric_SHAP.h5')
}

scaler_paths = {
    'target_BTC_Pric_Correlation': os.path.join(output_folder, 'target_BTC_Price_RobustScaler_target_BTC_Pric_Correlation.pkl'),
    'target_BTC_Pric_Granger': os.path.join(output_folder, 'target_BTC_Price_RobustScaler_target_BTC_Pric_Granger.pkl'),
    'target_BTC_Pric_RandomForest': os.path.join(output_folder, 'target_BTC_Price_RobustScaler_target_BTC_Pric_RandomForest.pkl'),
    'target_BTC_Pric_SHAP': os.path.join(output_folder, 'target_BTC_Price_RobustScaler_target_BTC_Pric_SHAP.pkl'),
    'Target_XRP_Pric_Correlation': os.path.join(output_folder, 'Target_XRP_Price_RobustScaler_Target_XRP_Pric_Correlation.pkl'),
    'Target_XRP_Pric_Granger': os.path.join(output_folder, 'Target_XRP_Price_RobustScaler_Target_XRP_Pric_Granger.pkl'),
    'Target_XRP_Pric_RandomForest': os.path.join(output_folder, 'Target_XRP_Price_RobustScaler_Target_XRP_Pric_RandomForest.pkl'),
    'Target_XRP_Pric_SHAP': os.path.join(output_folder, 'Target_XRP_Price_RobustScaler_Target_XRP_Pric_SHAP.pkl'),
    'Target_SOL_Pric_Correlation': os.path.join(output_folder, 'Target_SOL_Price_RobustScaler_Target_SOL_Pric_Correlation.pkl'),
    'Target_SOL_Pric_Granger': os.path.join(output_folder, 'Target_SOL_Price_RobustScaler_Target_SOL_Pric_Granger.pkl'),
    'Target_SOL_Pric_RandomForest': os.path.join(output_folder, 'Target_SOL_Price_RobustScaler_Target_SOL_Pric_RandomForest.pkl'),
    'Target_SOL_Pric_SHAP': os.path.join(output_folder, 'Target_SOL_Price_RobustScaler_Target_SOL_Pric_SHAP.pkl'),
    'Target_ETH_Pric_Correlation': os.path.join(output_folder, 'Target_ETH_Price_RobustScaler_Target_ETH_Pric_Correlation.pkl'),
    'Target_ETH_Pric_Granger': os.path.join(output_folder, 'Target_ETH_Price_RobustScaler_Target_ETH_Pric_Granger.pkl'),
    'Target_ETH_Pric_RandomForest': os.path.join(output_folder, 'Target_ETH_Price_RobustScaler_Target_ETH_Pric_RandomForest.pkl'),
    'Target_ETH_Pric_SHAP': os.path.join(output_folder, 'Target_ETH_Price_RobustScaler_Target_ETH_Pric_SHAP.pkl'),
    'Target_BNB_Pric_Correlation': os.path.join(output_folder, 'Target_BNB_Price_RobustScaler_Target_BNB_Pric_Correlation.pkl'),
    'Target_BNB_Pric_Granger': os.path.join(output_folder, 'Target_BNB_Price_RobustScaler_Target_BNB_Pric_Granger.pkl'),
    'Target_BNB_Pric_RandomForest': os.path.join(output_folder, 'Target_BNB_Price_RobustScaler_Target_BNB_Pric_RandomForest.pkl'),
    'Target_BNB_Pric_SHAP': os.path.join(output_folder, 'Target_BNB_Price_RobustScaler_Target_BNB_Pric_SHAP.pkl')
}

# Main function to iterate over models and make predictions
def main():
    predictions = []
    log = []

    for sheet_name in model_paths.keys():
        result = predict_with_model(sheet_name, model_paths[sheet_name], scaler_paths[sheet_name], log)
        if result:
            predictions.append(result)

    # Save predictions
    if predictions:
        df_predictions = pd.DataFrame(predictions)
        output_file = r'C:\omair\Projects\CapStone Project\Crypto Project results\predictions_Dense_Model.xlsx'
        df_predictions.to_excel(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    # Save log
    if log:
        df_log = pd.DataFrame(log)
        log_file = r'C:\omair\Projects\CapStone Project\Crypto Project results\prediction_Dense_log.xlsx'
        df_log.to_excel(log_file, index=False)
        print(f"Log saved to {log_file}")

if __name__ == "__main__":
    main()
