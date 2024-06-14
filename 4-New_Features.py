import pandas as pd
import pyodbc

# Database credentials
server = 'localhost'
database = 'CryptoPredictor'
username = 'sa'
password = 'Pakistan@12'

# Establishing the connection string
conn_str = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Table names from the previous list
historical_table_names = [
    "LINKUSDT_historical_data_futures",
    "LINKUSDT_historical_data_spot",
    "WBTCUSDT_historical_data_spot",
    "UNIUSDT_historical_data_futures",
    "UNIUSDT_historical_data_spot",
    "TRXUSDT_historical_data_futures",
    "TRXUSDT_historical_data_spot",
    "MATICUSDT_historical_data_futures",
    "MATICUSDT_historical_data_spot",
    "BTCUSDT_historical_data_futures",
    "BTCUSDT_historical_data_spot",
    "ETHUSDT_historical_data_futures",
    "ETHUSDT_historical_data_spot",
    "BNBUSDT_historical_data_futures",
    "BNBUSDT_historical_data_spot",
    "SOLUSDT_historical_data_futures",
    "SOLUSDT_historical_data_spot",
    "XRPUSDT_historical_data_futures",
    "XRPUSDT_historical_data_spot",
    "USDCUSDT_historical_data_futures",
    "USDCUSDT_historical_data_spot",
    "DOGEUSDT_historical_data_futures",
    "DOGEUSDT_historical_data_spot",
    "ADAUSDT_historical_data_futures",
    "ADAUSDT_historical_data_spot",
    "AVAXUSDT_historical_data_futures",
    "AVAXUSDT_historical_data_spot",
    "SHIBUSDT_historical_data_spot",
    "DOTUSDT_historical_data_futures",
    "DOTUSDT_historical_data_spot",
]

additional_table_names = [
    "Crude_Oil_Future_Prices",
    "GoldFutures",
]

stocks_table_1 = [
    "NYSE",
]
stocks_table_2 = [
    "NASDAQ",
]

# Connect to the database
with pyodbc.connect(conn_str) as conn:
    cursor = conn.cursor()

    # Alter historical tables to add net_price_change, price_variation, Target_net_price_change, and Target_price_variation columns
    for table in historical_table_names:
        alter_table_sql = f"""
        BEGIN TRY
            BEGIN TRANSACTION
                IF COL_LENGTH('{table}', 'net_price_change') IS NULL
                BEGIN
                    ALTER TABLE {table}
                    ADD net_price_change AS ([close] - [open]),
                    price_variation AS ([high] - [low]);
                END

                IF COL_LENGTH('{table}', 'Target_net_price_change') IS NULL
                BEGIN
                    ALTER TABLE {table}
                    ADD Target_net_price_change FLOAT,
                        Target_price_variation FLOAT;
                END
            COMMIT TRANSACTION
        END TRY
        BEGIN CATCH
            ROLLBACK TRANSACTION
            PRINT 'An error occurred with ' + '{table}'
            PRINT ERROR_MESSAGE()
        END CATCH
        """

        update_table_sql = f"""
        BEGIN TRY
            BEGIN TRANSACTION
                UPDATE {table}
                SET Target_net_price_change = (SELECT TOP 1 net_price_change
                                               FROM {table} AS next_day
                                               WHERE next_day.open_time > {table}.open_time
                                               ORDER BY next_day.open_time ASC),
                    Target_price_variation = (SELECT TOP 1 price_variation
                                              FROM {table} AS next_day
                                              WHERE next_day.open_time > {table}.open_time
                                              ORDER BY next_day.open_time ASC)
                WHERE Target_net_price_change IS NULL OR Target_price_variation IS NULL;
            COMMIT TRANSACTION
        END TRY
        BEGIN CATCH
            ROLLBACK TRANSACTION
            PRINT 'An error occurred with updating ' + '{table}'
            PRINT ERROR_MESSAGE()
        END CATCH
        """

        try:
            cursor.execute(alter_table_sql)
            conn.commit()
            print(f"Columns added for {table}")
        except Exception as e:
            print(f"An error occurred with {table}: {e}")

        try:
            cursor.execute(update_table_sql)
            conn.commit()
            print(f"Columns updated for {table}")
        except Exception as e:
            print(f"An error occurred with updating {table}: {e}")

    # Alter additional tables to add price_variation column
    for table in additional_table_names:
        alter_table_sql = f"""
        BEGIN TRY
            BEGIN TRANSACTION
                IF COL_LENGTH('{table}', 'price_variation') IS NULL
                BEGIN
                    ALTER TABLE {table}
                    ADD price_variation AS ([High] - [Low]);
                END
            COMMIT TRANSACTION
        END TRY
        BEGIN CATCH
            ROLLBACK TRANSACTION
            PRINT 'An error occurred with ' + '{table}'
            PRINT ERROR_MESSAGE()
        END CATCH
        """

        try:
            cursor.execute(alter_table_sql)
            conn.commit()
            print(f"Price variation column added for {table}")
        except Exception as e:
            print(f"An error occurred with {table}: {e}")

    for table in stocks_table_1:
        alter_table_sql = f"""
        BEGIN TRY
            BEGIN TRANSACTION
                IF COL_LENGTH('{table}', 'net_price_change') IS NULL
                BEGIN
                    ALTER TABLE {table}
                    ADD net_price_change AS ([close] - [open]),
                    price_variation AS ([high] - [low]);
                END
            COMMIT TRANSACTION
        END TRY
        BEGIN CATCH
            ROLLBACK TRANSACTION
            PRINT 'An error occurred with ' + '{table}'
            PRINT ERROR_MESSAGE()
        END CATCH
        """

        try:
            cursor.execute(alter_table_sql)
            conn.commit()
            print(f"Columns added and calculated for {table}")
        except Exception as e:
            print(f"An error occurred with {table}: {e}")

    for table in stocks_table_2:
        alter_table_sql = f"""
        BEGIN TRY
            BEGIN TRANSACTION
                IF COL_LENGTH('{table}', 'net_price_change') IS NULL
                BEGIN
                    ALTER TABLE {table}
                    ADD net_price_change AS ([Close/Last] - [open]),
                    price_variation AS ([high] - [low]);
                END
            COMMIT TRANSACTION
        END TRY
        BEGIN CATCH
            ROLLBACK TRANSACTION
            PRINT 'An error occurred with ' + '{table}'
            PRINT ERROR_MESSAGE()
        END CATCH
        """

        try:
            cursor.execute(alter_table_sql)
            conn.commit()
            print(f"Columns added and calculated for {table}")
        except Exception as e:
            print(f"An error occurred with {table}: {e}")

cursor.close()
conn.close()
