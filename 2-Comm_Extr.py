import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pyodbc

# Database connection parameters
server = 'localhost'
database = 'CryptoPredictor'  # replace with your database name
username = 'sa'
password = 'Pakistan@12'  # replace with your password
driver = 'ODBC Driver 17 for SQL Server'

# Define symbols for the commodities in Yahoo Finance
symbols = {
    'Crude_Oil_Future_Prices': 'CL=F',
    'Crude_Oil_Spot_Prices': 'USO',  # Using ETF as a proxy for spot price
    'gold_spot_prices': 'GLD',  # Using ETF as a proxy for spot price
    'GoldFutures': 'GC=F',
}

# Establishing the connection
print("Connecting to the SQL Server database...")
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
print("Successfully connected to the database.")

# Function to get the last date data is available for each commodity table
def get_last_date(table_name):
    # Determine the correct date column based on the table
    if 'Crude_Oil' in table_name:
        date_column = 'observation_date'
    elif 'gold_spot_prices' in table_name:
        date_column = 'date'
    elif 'GoldFutures' in table_name:
        date_column = 'Date'
    else:
        raise Exception(f"Unknown table: {table_name}")

    query = f"SELECT MAX([{date_column}]) FROM {table_name}"
    print(f"Fetching the last available date from {table_name} using column {date_column}...")
    with conn.cursor() as cursor:
        cursor.execute(query)
        last_date = cursor.fetchone()[0]
        print(f"Last available date for {table_name}: {last_date}")
        return last_date if last_date else datetime(2000, 1, 1)

# Function to save the dataframe to the SQL table
def save_to_sql(df, table_name, column_mapping):
    print(f"Saving data to {table_name}...")
    sql_columns = ', '.join([f"[{col}]" for col in column_mapping.values()])
    placeholders = ', '.join('?' for _ in column_mapping.keys())
    sql = f"INSERT INTO {table_name} ({sql_columns}) VALUES ({placeholders})"

    with conn.cursor() as cursor:
        for row in df.itertuples(index=False, name=None):
            cursor.execute(sql, row)
        conn.commit()
    print(f"Data saved to {table_name} successfully.")

# Download and save the data to SQL
for name, symbol in symbols.items():
    table_name = f"dbo.{name}"
    last_date = get_last_date(table_name)
    end_date = datetime.now()

    print(f"Downloading data for {name} from {last_date + timedelta(days=1)} to {end_date}...")
    data = yf.download(symbol, start=last_date + timedelta(days=1), end=end_date)

    # Data preprocessing
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'observation_date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'price', 'Volume': 'volume'}, inplace=True)

    # Setup column mapping
    if 'Crude_Oil_Spot_Prices' in table_name:  # For spot prices table
        column_mapping = {
            'observation_date': 'observation_date',
            'price': 'price'
        }
        df = data[['observation_date', 'price']]  # Select only the observation_date and price columns
    elif 'Crude_Oil_Future_Prices' in table_name:  # For future prices table
        column_mapping = {
            'observation_date': 'observation_date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'price': 'price',
            'volume': 'volume'
        }
        df = data[list(column_mapping.keys())]  # Select all relevant columns
    elif 'gold_spot_prices' in name:
        column_mapping = {
            'observation_date': 'date',  # Make sure 'date' is the correct column name in the SQL table
            'price': 'price'
        }
    elif 'GoldFutures' in name:
        column_mapping = {
            'observation_date': 'Date',
            'price': 'Price',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        }

    # Apply renaming based on column mapping
    df = data[list(column_mapping.keys())]
    df.columns = [column_mapping[col] for col in df.columns]

    # Save to SQL table
    save_to_sql(df, table_name, column_mapping)

# Close the database connection
conn.close()
print("All tasks completed successfully.")
