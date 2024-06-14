import pandas as pd
import pyodbc
from datetime import datetime, timedelta

# Database connection parameters
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=CryptoPredictor;"
    "UID=sa;"
    "PWD=Pakistan@12"
)

# Define the columns mapping from source to target
columns_mapping = {
    'close_BTCUSDT_historical_data_spot': 'Target_BTC_Price',
    'close_XRPUSDT_historical_data_spot': 'Target_XRP_Price',
    'close_SOLUSDT_historical_data_spot': 'Target_SOL_Price',
    'close_ETHUSDT_historical_data_spot': 'Target_ETH_Price',
    'close_BNBUSDT_historical_data_spot': 'Target_BNB_Price',
    'high_BTCUSDT_historical_data_spot': 'Target_High_BTC',
    'high_XRPUSDT_historical_data_spot': 'Target_High_XRP',
    'high_SOLUSDT_historical_data_spot': 'Target_High_SOL',
    'high_ETHUSDT_historical_data_spot': 'Target_High_ETH',
    'high_BNBUSDT_historical_data_spot': 'Target_High_BNB',
    'low_BTCUSDT_historical_data_spot': 'Target_Low_BTC',
    'low_XRPUSDT_historical_data_spot': 'Target_Low_XRP',
    'low_SOLUSDT_historical_data_spot': 'Target_Low_SOL',
    'low_ETHUSDT_historical_data_spot': 'Target_Low_ETH',
    'low_BNBUSDT_historical_data_spot': 'Target_Low_BNB'
}

# Establish connection to the SQL Server database
conn = pyodbc.connect(conn_str)
print("Database connection established.")

# Process each column pair
for source_col, target_col in columns_mapping.items():
    print(f"Processing: {source_col} to {target_col}")
    cursor = conn.cursor()

    # Check if the target column exists
    column_check_query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Data' AND COLUMN_NAME = '{target_col}'"
    cursor.execute(column_check_query)
    if not cursor.fetchone():
        # Add the target column if it does not exist
        cursor.execute(f"ALTER TABLE Data ADD {target_col} FLOAT")
        conn.commit()
        print(f"Added column {target_col} to database.")

    # Fetch data from the database using pandas directly
    data_query = f"SELECT Date, {source_col} FROM Data"
    data = pd.read_sql(data_query, conn)
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)

    # Create the target column by shifting the source column by -1 to get the next date's data
    next_day_data = data.set_index('Date').shift(-1).reset_index()
    data[target_col] = next_day_data[source_col]

    # Filter out rows where target column is NaN
    data = data.dropna(subset=[target_col])
    if data.empty:
        print(f"No valid data found for {source_col}. Skipping.")
        continue

    # Update the database with the new column values
    update_sql = f"UPDATE Data SET {target_col} = ? WHERE Date = ?"
    updates_count = 0
    for index, row in data.iterrows():
        # Only update if the target value is not NaN
        if pd.notna(row[target_col]):
            cursor.execute(update_sql, float(row[target_col]), row['Date'])
            updates_count += 1

    cursor.close()
    print(f"Updated {updates_count} rows for column {target_col}.")

# Commit changes and close the connection
conn.commit()
conn.close()
print("Database updates completed successfully. Connection closed.")
