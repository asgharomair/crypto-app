import pandas as pd
import numpy as np
import pyodbc
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection parameters
server = 'localhost'
database = 'CryptoPredictor'
username = 'sa'
password = 'Pakistan@12'
driver = '{ODBC Driver 17 for SQL Server}'

try:
    # Establish connection to the SQL Server database
    conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    conn = pyodbc.connect(conn_str)
    logging.info("Connected to the database successfully.")

    # SQL query to extract the 'Date' column from the 'Data' table
    query = 'SELECT * FROM Data'

    # Read the data into a pandas DataFrame
    df = pd.read_sql(query, conn)
    logging.info("Data successfully read from the database.")

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    logging.info("Converted 'Date' column to datetime format.")

    # Compute date-related features directly from 'Date'
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] > 4).astype(int)
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['IsQuarterStart'] = df['Date'].dt.is_quarter_start.astype(int)
    df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end.astype(int)
    df['IsYearStart'] = df['Date'].dt.is_year_start.astype(int)
    df['IsYearEnd'] = df['Date'].dt.is_year_end.astype(int)
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['sin_day'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['cos_day'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Update the database with new columns
    cursor = conn.cursor()
    for index, row in df.iterrows():
        try:
            # Prepare the SQL UPDATE statement with placeholders for each column value
            update_query = f"""
            UPDATE Data SET 
            Year = ?, Month = ?, Day = ?, WeekOfYear = ?, DayOfWeek = ?, IsWeekend = ?, 
            IsMonthStart = ?, IsMonthEnd = ?, Quarter = ?, IsQuarterStart = ?, IsQuarterEnd = ?, 
            IsYearStart = ?, IsYearEnd = ?, sin_month = ?, cos_month = ?, sin_day = ?, cos_day = ?, 
            sin_dayofweek = ?, cos_dayofweek = ?
            WHERE Date = ?
            """
            # Execute the update with values from the current row
            cursor.execute(update_query,
                           row['Year'], row['Month'], row['Day'],
                           row['WeekOfYear'], row['DayOfWeek'], row['IsWeekend'],
                           row['IsMonthStart'], row['IsMonthEnd'], row['Quarter'],
                           row['IsQuarterStart'], row['IsQuarterEnd'],
                           row['IsYearStart'], row['IsYearEnd'],
                           row['sin_month'], row['cos_month'],
                           row['sin_day'], row['cos_day'],
                           row['sin_dayofweek'], row['cos_dayofweek'],
                           row['Date'])
            logging.info(f"Updated row with Date {row['Date']} in the database.")
        except Exception as e:
            logging.error(f"Failed to update row with Date {row['Date']}: {str(e)}")

    # Commit the changes to the database
    conn.commit()
    logging.info("All changes committed to the database.")

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
finally:
    # Ensure the connection is closed
    conn.close()
    logging.info("Database connection closed.")
