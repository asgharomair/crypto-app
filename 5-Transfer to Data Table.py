import pyodbc
import pandas as pd
from datetime import datetime, timedelta

# Establish database connection
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=CryptoPredictor;"
    "UID=sa;"
    "PWD=Pakistan@12"
)
cursor = conn.cursor()

def create_data_table():
    print("Checking if 'Data' table exists and creating if not...")
    cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'Data')
        CREATE TABLE Data (Date DATETIME PRIMARY KEY);
    """)
    conn.commit()
    print("Table 'Data' is ready.")

def update_date_column():
    print("Updating 'Date' column in 'Data' table...")
    cursor.execute("SELECT MAX(Date) FROM Data;")
    last_date = cursor.fetchone()[0]
    if last_date is None:
        last_date = datetime(2018, 1, 1)  # Start from a specific date
    else:
        last_date += timedelta(days=1)

    end_date = datetime.now()
    while last_date <= end_date:
        cursor.execute("INSERT INTO Data (Date) VALUES (?);", last_date)
        last_date += timedelta(days=1)
    conn.commit()
    print("Date column updated up to the current date.")

def read_date_column_info():
    print("Reading date column information from CSV...")
    df = pd.read_csv("C:/omair/Projects/CapStone Project/Crypto Project results/Date Table.csv")
    print("Date column information loaded.")
    return df

def update_data_table_schema():
    print("Updating 'Data' table schema...")
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME != 'Data' AND TABLE_CATALOG = 'CryptoPredictor'
    """)
    columns = cursor.fetchall()

    # Check and add columns with the appropriate type
    for table, column in columns:
        new_column_name = f"{column}_{table}"
        cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'Data' AND COLUMN_NAME = ?;
        """, new_column_name)
        if not cursor.fetchone():  # If the column does not exist, then add it
            cursor.execute(f"ALTER TABLE Data ADD [{new_column_name}] FLOAT;")
            conn.commit()
    print("Schema updated with float columns.")
def copy_data_to_data_table(date_column_info):
    print("Copying data to 'Data' table...")
    for index, row in date_column_info.iterrows():
        table_name = row['TableName']
        date_column = row['ColumnName']
        cursor.execute(f"""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'
        """)
        column_names = [col[0] for col in cursor.fetchall()]

        update_query = "UPDATE D SET " + ", ".join(
            [f"D.[{col}_{table_name}] = CAST(T.[{col}] AS FLOAT)" for col in column_names if col != date_column]
        ) + f" FROM Data D JOIN {table_name} T ON D.Date = T.[{date_column}] WHERE D.Date IS NOT NULL;"

        try:
            cursor.execute(update_query)
            conn.commit()
            print(f"Data successfully copied from table {table_name}.")
        except Exception as e:
            print(f"Error copying data from {table_name}: {str(e)}")

def main():
    create_data_table()
    update_date_column()
    date_column_info = read_date_column_info()
    update_data_table_schema()
    copy_data_to_data_table(date_column_info)
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
