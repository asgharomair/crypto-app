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
conn = pyodbc.connect(conn_string)

# Get list of tables
cursor = conn.cursor()
cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
tables = cursor.fetchall()

# Data collection
data_frames = []

for table in tables:
    table_name = table[0]
    cursor.execute(f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE
        FROM
            INFORMATION_SCHEMA.COLUMNS
        WHERE
            TABLE_NAME = ?
    """, table_name)
    columns = cursor.fetchall()

    for column in columns:
        column_name, data_type = column
        try:
            # Note: Ensure that the column name is properly escaped to handle special characters or reserved keywords.
            sql_query = f"""
                SELECT
                    MAX([{column_name}]) AS MaxValue,
                    MIN([{column_name}]) AS MinValue
                FROM
                    [{table_name}]
            """
            cursor.execute(sql_query)
            max_min = cursor.fetchone()
            max_value, min_value = max_min
        except Exception as e:
            max_value, min_value = None, None
            print(f"Error retrieving max/min for {column_name} in {table_name}: {e}")

        # Attempt to find primary key status
        cursor.execute(f"""
            SELECT
                KCU.COLUMN_NAME
            FROM
                INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KCU
            WHERE
                KCU.TABLE_NAME = ? AND KCU.COLUMN_NAME = ?
        """, table_name, column_name)
        pk_info = cursor.fetchone()
        is_pk = 'PK' if pk_info else None

        # Create a DataFrame from fetched data
        df = pd.DataFrame({
            "TableName": [table_name],
            "ColumnName": [column_name],
            "DataType": [data_type],
            "PrimaryKey": [is_pk],
            "MaxValue": [max_value],
            "MinValue": [min_value]
        })
        data_frames.append(df)

# Concatenate all dataframes into one
final_df = pd.concat(data_frames, ignore_index=True)

# Save to CSV
output_path = r'C:\omair\Projects\CapStone Project\Crypto Project results\database_info.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
final_df.to_csv(output_path, index=False)

print("Data extraction complete and saved to CSV.")
