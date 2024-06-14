import yfinance as yf
import pandas as pd
import pyodbc
from datetime import datetime, timedelta

# Database connection parameters
server = 'localhost'
database = 'CryptoPredictor'
username = 'sa'
password = 'Pakistan@12'
driver = 'ODBC Driver 17 for SQL Server'

# Establishing the connection
conn_str = (
    f'DRIVER={driver};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)
conn = pyodbc.connect(conn_str)

# Function to get the last date data is available for each index
def get_max_date(table_name):
    query = f"SELECT MAX([Date]) FROM {table_name}"
    with conn.cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
    return result[0] if result else None

# Function to insert DataFrame into SQL table
def insert_dataframe_to_sql(df, table_name):
    columns = [f"[{col}]" for col in df.columns]  # enclose column names in brackets
    placeholders = ', '.join('?' for _ in columns)
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    with conn.cursor() as cursor:
        for index, row in df.iterrows():
            try:
                cursor.execute(sql, tuple(row))
                conn.commit()
                print(f"Inserted row {index+1} of {len(df)} into {table_name}")
            except Exception as e:
                print(f"Error inserting row {index+1}: {e}")
                conn.rollback()  # Rollback if there is any error

# Fetch the last date data is available for each index
print("Fetching the last date from the database for each index...")
max_date_nasdaq = get_max_date('dbo.NASDAQ')
max_date_nyse = get_max_date('dbo.NYSE')
print(f"Last date for NASDAQ: {max_date_nasdaq}")
print(f"Last date for NYSE: {max_date_nyse}")

# Define the new start date for each index
start_date_nasdaq = (max_date_nasdaq + timedelta(days=1)) if max_date_nasdaq else datetime(2000, 1, 1)
start_date_nyse = (max_date_nyse + timedelta(days=1)) if max_date_nyse else datetime(2000, 1, 1)

# Download new data from Yahoo Finance for NASDAQ and NYSE
print("Downloading new data from Yahoo Finance for NASDAQ...")
nasdaq_data = yf.download('^IXIC', start=start_date_nasdaq, end=datetime.now())
print("Downloading new data from Yahoo Finance for NYSE...")
nyse_data = yf.download('^NYA', start=start_date_nyse, end=datetime.now())

# Format the data to match the table schema and adjust column names
nasdaq_data = nasdaq_data.reset_index()
nyse_data = nyse_data.reset_index()
nasdaq_data.rename(columns={'Close': 'Close/Last'}, inplace=True)  # renaming to match the NASDAQ table schema

# Convert Date to the required format for SQL
nasdaq_data['Date'] = nasdaq_data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
nyse_data['Date'] = nyse_data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Insert the new data into SQL Server
if not nasdaq_data.empty:
    print("Inserting new NASDAQ data into the database...")
    insert_dataframe_to_sql(nasdaq_data[['Date', 'Open', 'High', 'Low', 'Close/Last']], 'dbo.NASDAQ')

if not nyse_data.empty:
    print("Inserting new NYSE data into the database...")
    insert_dataframe_to_sql(nyse_data[['Date', 'Open', 'High', 'Low', 'Close']], 'dbo.NYSE')

print("Update complete!")

# Close the database connection
conn.close()
