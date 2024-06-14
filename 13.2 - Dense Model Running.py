import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pyodbc
from datetime import datetime
from sklearn.preprocessing import RobustScaler
import joblib  # To save/load the scaler

# Database credentials
server = os.getenv('DB_SERVER', 'localhost')
database = os.getenv('DB_NAME', 'CryptoPredictor')
username = os.getenv('DB_USER', 'sa')
password = os.getenv('DB_PASSWORD', 'Pakistan@12')

# Selected features based on correlation analysis
selected_features = [
    'close_BTCUSDT_historical_data_spot',
    'high_BTCUSDT_historical_data_spot',
    'low_BTCUSDT_historical_data_spot',
    'BTCUSDT_EMA5',
    'open_BTCUSDT_historical_data_spot',
    'close_WBTCUSDT_historical_data_spot',
    'close_BTCUSDT_historical_data_futures',
    'high_WBTCUSDT_historical_data_spot',
    'low_WBTCUSDT_historical_data_spot',
    'high_BTCUSDT_historical_data_futures',
    'BTCUSDT_EMA10',
    'low_BTCUSDT_historical_data_futures',
    'open_WBTCUSDT_historical_data_spot',
    'open_BTCUSDT_historical_data_futures',
    'Target_ETH_Price',
    'close_ETHUSDT_historical_data_spot',
    'high_ETHUSDT_historical_data_spot',
    'low_ETHUSDT_historical_data_spot',
    'open_ETHUSDT_historical_data_spot',
    'close_ETHUSDT_historical_data_futures',
    'low_ETHUSDT_historical_data_futures',
    'high_ETHUSDT_historical_data_futures',
    'High_NASDAQ',
    'Open_NASDAQ',
    'Low_NASDAQ',
    'Close/Last_NASDAQ',
    'open_ETHUSDT_historical_data_futures',
    'Target_BNB_Price',
    'high_BNBUSDT_historical_data_spot',
    'close_BNBUSDT_historical_data_spot',
    'High_NYSE',
    'close_time_WBTCUSDT_historical_data_spot',
    'open_BNBUSDT_historical_data_spot',
    'Close_NYSE',
    'low_BNBUSDT_historical_data_spot',
    'Open_NYSE',
    'Low_NYSE',
    'close_SOLUSDT_historical_data_futures',
    'low_SOLUSDT_historical_data_futures',
    'high_SOLUSDT_historical_data_futures',
    'open_SOLUSDT_historical_data_futures',
    'high_TRXUSDT_historical_data_spot',
    'close_TRXUSDT_historical_data_spot',
    'open_TRXUSDT_historical_data_spot',
    'low_TRXUSDT_historical_data_spot',
    'close_time_USDCUSDT_historical_data_futures',
    'low_LINKUSDT_historical_data_spot',
    'close_LINKUSDT_historical_data_spot',
    'open_LINKUSDT_historical_data_spot',
    'high_LINKUSDT_historical_data_spot',
    'low_LINKUSDT_historical_data_futures',
    'close_LINKUSDT_historical_data_futures',
    'close_BNBUSDT_historical_data_futures',
    'open_LINKUSDT_historical_data_futures',
    'high_BNBUSDT_historical_data_futures',
    'high_LINKUSDT_historical_data_futures',
    'low_BNBUSDT_historical_data_futures',
    'open_BNBUSDT_historical_data_futures',
    'high_TRXUSDT_historical_data_futures',
    'close_TRXUSDT_historical_data_futures',
    'open_TRXUSDT_historical_data_futures',
    'low_ADAUSDT_historical_data_spot',
    'close_ADAUSDT_historical_data_spot',
    'low_TRXUSDT_historical_data_futures',
    'open_ADAUSDT_historical_data_spot',
    'high_ADAUSDT_historical_data_spot',
    'Target_SOL_Price',
    'close_SOLUSDT_historical_data_spot',
    'high_SOLUSDT_historical_data_spot',
    'low_SOLUSDT_historical_data_spot',
    'open_SOLUSDT_historical_data_spot',
    'Target_XRP_Price',
    'low_XRPUSDT_historical_data_spot',
    'close_XRPUSDT_historical_data_spot',
    'open_XRPUSDT_historical_data_spot',
    'close_DOGEUSDT_historical_data_futures',
    'high_XRPUSDT_historical_data_spot',
    'high_DOGEUSDT_historical_data_futures',
    'low_DOGEUSDT_historical_data_spot',
    'open_DOGEUSDT_historical_data_futures',
    'low_DOGEUSDT_historical_data_futures',
    'Low_GoldFutures',
    'Price_GoldFutures',
    'Open_GoldFutures',
    'High_GoldFutures',
    'close_DOGEUSDT_historical_data_spot',
    'open_DOGEUSDT_historical_data_spot',
    'price_variation_BTCUSDT_historical_data_spot',
    'number_of_trades_WBTCUSDT_historical_data_spot',
    'close_XRPUSDT_historical_data_futures',
    'low_XRPUSDT_historical_data_futures',
    'open_XRPUSDT_historical_data_futures',
    'Year',
    'high_DOGEUSDT_historical_data_spot',
    'high_XRPUSDT_historical_data_futures',
    'close_time_BNBUSDT_historical_data_spot',
    'close_time_ETHUSDT_historical_data_spot',
    'close_time_BTCUSDT_historical_data_spot',
    'close_time_USDCUSDT_historical_data_spot',
    'low_DOTUSDT_historical_data_spot',
    'close_time_ADAUSDT_historical_data_spot',
    'close_DOTUSDT_historical_data_spot',
    'close_time_XRPUSDT_historical_data_spot',
    'number_of_trades_ETHUSDT_historical_data_spot',
    'open_DOTUSDT_historical_data_spot',
    'high_DOTUSDT_historical_data_spot',
    'close_time_TRXUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_ETHUSDT_historical_data_spot',
    'price_variation_SOLUSDT_historical_data_futures',
    'quote_asset_volume_ETHUSDT_historical_data_spot',
    'price_variation_WBTCUSDT_historical_data_spot',
    'close_MATICUSDT_historical_data_spot',
    'price_variation_BNBUSDT_historical_data_spot',
    'low_UNIUSDT_historical_data_spot',
    'high_MATICUSDT_historical_data_spot',
    'low_MATICUSDT_historical_data_spot',
    'open_MATICUSDT_historical_data_spot',
    'close_UNIUSDT_historical_data_spot',
    'open_UNIUSDT_historical_data_spot',
    'price_variation_ETHUSDT_historical_data_spot',
    'close_AVAXUSDT_historical_data_futures',
    'high_UNIUSDT_historical_data_spot',
    'high_AVAXUSDT_historical_data_futures',
    'close_SHIBUSDT_historical_data_spot',
    'low_AVAXUSDT_historical_data_futures',
    'number_of_trades_BNBUSDT_historical_data_spot',
    'open_AVAXUSDT_historical_data_futures',
    'high_SHIBUSDT_historical_data_spot',
    'open_SHIBUSDT_historical_data_spot',
    'low_SHIBUSDT_historical_data_spot',
    'high_AVAXUSDT_historical_data_spot',
    'close_AVAXUSDT_historical_data_spot',
    'close_UNIUSDT_historical_data_futures',
    'open_AVAXUSDT_historical_data_spot',
    'open_UNIUSDT_historical_data_futures',
    'high_UNIUSDT_historical_data_futures',
    'close_time_LINKUSDT_historical_data_spot',
    'low_UNIUSDT_historical_data_futures',
    'low_AVAXUSDT_historical_data_spot',
    'close_DOTUSDT_historical_data_futures',
    'low_DOTUSDT_historical_data_futures',
    'high_DOTUSDT_historical_data_futures',
    'open_DOTUSDT_historical_data_futures',
    'price_variation_SOLUSDT_historical_data_spot',
    'price_variation_BTCUSDT_historical_data_futures',
    'quote_asset_volume_BNBUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_BNBUSDT_historical_data_spot',
    'close_ADAUSDT_historical_data_futures',
    'low_ADAUSDT_historical_data_futures',
    'open_ADAUSDT_historical_data_futures',
    'high_ADAUSDT_historical_data_futures',
    'taker_buy_base_asset_volume_USDCUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_USDCUSDT_historical_data_spot',
    'close_time_MATICUSDT_historical_data_spot',
    'number_of_trades_SHIBUSDT_historical_data_spot',
    'volume_USDCUSDT_historical_data_spot',
    'quote_asset_volume_USDCUSDT_historical_data_spot',
    'price_variation_ADAUSDT_historical_data_spot',
    'quote_asset_volume_SOLUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_SOLUSDT_historical_data_spot',
    'price_variation_AVAXUSDT_historical_data_spot',
    'price_variation_AVAXUSDT_historical_data_futures',
    'number_of_trades_LINKUSDT_historical_data_spot',
    'number_of_trades_TRXUSDT_historical_data_spot',
    'number_of_trades_ADAUSDT_historical_data_spot',
    'close_time_DOGEUSDT_historical_data_spot',
    'number_of_trades_USDCUSDT_historical_data_spot',
    'quote_asset_volume_WBTCUSDT_historical_data_spot',
    'price_variation_LINKUSDT_historical_data_futures',
    'number_of_trades_SOLUSDT_historical_data_spot',
    'number_of_trades_XRPUSDT_historical_data_spot',
    'price_variation_ETHUSDT_historical_data_futures',
    'quote_asset_volume_TRXUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_TRXUSDT_historical_data_spot',
    'price_variation_LINKUSDT_historical_data_spot',
    'quote_asset_volume_SOLUSDT_historical_data_futures',
    'quote_asset_volume_XRPUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_SOLUSDT_historical_data_futures',
    'taker_buy_quote_asset_volume_XRPUSDT_historical_data_spot'
]

# Function to create a database connection
def create_connection():
    try:
        conn = pyodbc.connect(
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )
        print("Database connection established.")
        return conn
    except pyodbc.Error as e:
        print(f"Error in connection: {e}")
        return None

# Function to get the latest available date from the database
def get_latest_date(conn):
    try:
        query = """
        SELECT MAX([Date]) AS latest_date
        FROM [Data]
        """
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()

        if not row or not row[0]:
            raise ValueError("No data found in the database.")

        return row[0]
    except Exception as e:
        print(f"Error fetching latest date: {e}")
        return None

# Function to load data for a specific date
def load_data_for_date(conn, specific_date):
    try:
        # Format date as string in 'YYYY-MM-DD' format
        formatted_date = specific_date.strftime('%Y-%m-%d')
        query = f"""
        SELECT {', '.join([f'[{feature}]' for feature in selected_features])}
        FROM [Data]
        WHERE [Date] = '{formatted_date}'
        """
        data = pd.read_sql(query, conn)

        if data.empty:
            raise ValueError(f"Data for {specific_date} is incomplete or not found.")

        return data
    except Exception as e:
        print(f"Error loading data for prediction: {e}")
        return None

# Main function to perform the steps
def main():
    conn = create_connection()
    if conn is None:
        return

    # Get the latest available date
    latest_date = get_latest_date(conn)
    if latest_date is None:
        print("No latest date found.")
        return

    # Load values from the database for the latest date
    data = load_data_for_date(conn, latest_date)
    if data is None or data.empty:
        print("No data found for the latest date.")
        return

    # Drop the 'Date' column if it exists in the data
    if 'Date' in data.columns:
        data = data.drop(columns=['Date'])

    # Prepare the data for prediction using these values
    data_values = data.values  # Convert to numpy array

    # Load the fitted scaler
    scaler_path = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\robust_scaler.pkl'
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        print(f"Scaler file not found at {scaler_path}. Ensure the file exists.")
        return

    # Scale the data using the pre-fitted scaler
    X_scaled = scaler.transform(data_values)

    # Load the trained model
    model_path = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\BTC_Dense_Model.h5'
    model = tf.keras.models.load_model(model_path)

    # Predict the value using the model
    prediction = model.predict(X_scaled)
    print(f"Predicted BTC Price for {latest_date}: {prediction[0][0]}")

if __name__ == "__main__":
    main()
