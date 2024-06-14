import pyodbc
import pandas as pd

# Establish database connection
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=CryptoPredictor;"
    "UID=sa;"
    "PWD=Pakistan@12"
)
cursor = conn.cursor()

# Query to get all tables and their datetime columns
query = """
SELECT 
    TABLE_NAME, 
    COLUMN_NAME, 
    DATA_TYPE,
    COLUMNPROPERTY(object_id(TABLE_NAME), COLUMN_NAME, 'IsPrimaryKey') AS PrimaryKey
FROM INFORMATION_SCHEMA.COLUMNS
WHERE DATA_TYPE = 'datetime';
"""
cursor.execute(query)
columns_info = cursor.fetchall()

# Preparing the DataFrame
data = []
for table, column, datatype, primary_key in columns_info:
    # Query to get max and min values
    cursor.execute(f"SELECT MAX([{column}]), MIN([{column}]) FROM [{table}];")
    max_value, min_value = cursor.fetchone()
    data.append({
        "TableName": table,
        "ColumnName": column,
        "DataType": datatype,
        "PrimaryKey": 'Yes' if primary_key else 'No',
        "MaxValue": max_value,
        "MinValue": min_value
    })

df = pd.DataFrame(data, columns=["TableName", "ColumnName", "DataType", "PrimaryKey", "MaxValue", "MinValue"])

# Save DataFrame to CSV
csv_path = 'C:/omair/Projects/CapStone Project/Crypto Project results/Date Table.csv'
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully at {csv_path}.")
