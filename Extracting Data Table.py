import pyodbc
import pandas as pd
import os

# Database connection parameters
conn_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=CryptoPredictor;"
    "UID=sa;"
    "PWD=Pakistan@12"
)

# Connect to the database
try:
    conn = pyodbc.connect(conn_string)
    print("Database connection successful.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

# SQL query to fetch all data from the table 'Data'
query = "SELECT * FROM Data"

# Use pandas to load the query results into a DataFrame
try:
    data_df = pd.read_sql(query, conn)
    print("Data read successfully from the database.")
except Exception as e:
    print(f"Error reading data from the database: {e}")
    conn.close()
    exit()

# Close the database connection
conn.close()
print("Database connection closed.")

# Check if DataFrame is empty
if data_df.empty:
    print("No data found in the table 'Data'.")
    exit()

# Print DataFrame information for debugging
print(data_df.info())

# Output path for the CSV file
output_path = r'C:\omair\Projects\CapStone Project\Crypto Project results\Data_Table.csv'

# Print the intended output path for verification
print(f"Intended output path: {output_path}")

# Create directory if it does not exist
try:
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Directory {os.path.dirname(output_path)} created.")
    else:
        print(f"Directory {os.path.dirname(output_path)} already exists.")
except Exception as e:
    print(f"Error creating directory: {e}")
    exit()

# Save the DataFrame to CSV
try:
    data_df.to_csv(output_path, index=False)
    print(f"Data has been successfully saved to {output_path}")
except Exception as e:
    print(f"Error saving DataFrame to CSV: {e}")
    exit()

# Verify if the file has been created
if os.path.isfile(output_path):
    print(f"File {output_path} has been created successfully.")
else:
    print(f"File {output_path} has not been created.")
