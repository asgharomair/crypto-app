import pandas as pd
import numpy as np
import pyodbc

# Establish database connection
print("Connecting to database...")
conn = pyodbc.connect(
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=localhost;'
    r'DATABASE=CryptoPredictor;'
    r'UID=sa;'
    r'PWD=Pakistan@12'
)
cursor = conn.cursor()
print("Database connection established.")

# Extract data from the database
print("Executing SQL query...")
sql_query = 'SELECT Date, close_BTCUSDT_historical_data_spot FROM Data'
data = pd.read_sql(sql_query, conn)
print("Data retrieved from database.")

# Ensure correct data types and handle NaNs
print("Processing data types and handling NaNs...")
data['close_BTCUSDT_historical_data_spot'] = data['close_BTCUSDT_historical_data_spot'].astype(float).fillna(0)

# Calculate new features with NaN handling
print("Calculating new features...")
data['BTCUSDT_EMA5'] = data['close_BTCUSDT_historical_data_spot'].ewm(span=5, adjust=False).mean().fillna(0)
data['BTCUSDT_EMA10'] = data['close_BTCUSDT_historical_data_spot'].ewm(span=10, adjust=False).mean().fillna(0)
data['BTCUSDT_DailyPctChange'] = data['close_BTCUSDT_historical_data_spot'].pct_change().fillna(0) * 100

delta = data['close_BTCUSDT_historical_data_spot'].diff().fillna(0)
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=5).mean().fillna(0)
avg_loss = loss.rolling(window=5).mean().fillna(0)
rs = avg_gain / avg_loss
data['BTCUSDT_RSI5'] = (100 - (100 / (1 + rs))).fillna(0)
data['BTCUSDT_Volatility5d'] = data['BTCUSDT_DailyPctChange'].rolling(window=5).std().fillna(0)
data['BTCUSDT_Volatility10d'] = data['BTCUSDT_DailyPctChange'].rolling(window=10).std().fillna(0)
print("Feature calculation completed.")

# Function to update or create new feature columns in the database
def update_or_create_features(data, feature_name, cursor):
    print(f"Checking if column {feature_name} exists in database...")
    cursor.execute(
        f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Data' AND COLUMN_NAME = '{feature_name}'")
    if not cursor.fetchone():
        print(f"Column {feature_name} does not exist, creating column...")
        cursor.execute(f"ALTER TABLE Data ADD {feature_name} FLOAT")
        conn.commit()
        print(f"Column {feature_name} created.")

    print(f"Preparing to update {feature_name} in database...")
    update_query = f"UPDATE Data SET {feature_name} = ? WHERE Date = ?"
    insert_query = f"INSERT INTO Data (Date, {feature_name}) VALUES (?, ?)"

    # Initialize counters
    update_count = 0
    insert_count = 0

    for index, row in data.iterrows():
        # Check if the date exists in the database
        cursor.execute("SELECT Date FROM Data WHERE Date = ?", (row['Date'],))
        if cursor.fetchone():
            # Update existing record
            cursor.execute(update_query, (float(row[feature_name]), row['Date']))
            update_count += 1
        else:
            # Insert new record
            cursor.execute(insert_query, (row['Date'], float(row[feature_name])))
            insert_count += 1

        # Output progress every 100 records
        if (update_count + insert_count) % 100 == 0:
            print(f"Processed {update_count + insert_count} records...")

    conn.commit()
    print(f"Updated {update_count} records and inserted {insert_count} new records for {feature_name}.")

# Update/Create new feature columns
features = ['BTCUSDT_EMA5', 'BTCUSDT_EMA10', 'BTCUSDT_DailyPctChange', 'BTCUSDT_RSI5', 'BTCUSDT_Volatility5d', 'BTCUSDT_Volatility10d']
for feature in features:
    update_or_create_features(data, feature, cursor)

cursor.close()
conn.close()
print("Database connection closed.")
