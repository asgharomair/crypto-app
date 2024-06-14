import pyodbc
from binance.client import Client
import pandas as pd
from datetime import datetime
import time

# Binance API Credentials
api_key = 'O9u3cUeRrGtbj0vNExbYdWMbLd2JQ3HKsLgs3I7LM6iRBzLRLrcS69D9nTTQvpP7'
api_secret = 'IG092opciS2ggLek4GGST8KLH2xZYTc9VwLnvVlKpN0KV8KkOI2bzu2EM9L6iqg5'
client = Client(api_key, api_secret)

# Database connection string
conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=CryptoPredictor;UID=sa;PWD=Pakistan@12'

def check_table_exists(symbol, is_futures=False):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    table_suffix = "futures" if is_futures else "spot"
    table_name = f"{symbol}_historical_data_{table_suffix}"
    cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", table_name)
    result = cursor.fetchone()
    conn.close()
    return result is not None, table_name

def fetch_klines(symbol, interval, start_str, end_str=None, is_futures=False):
    fetched_data = []
    limit = 1000
    while True:
        if is_futures:
            klines = client.futures_klines(symbol=symbol, interval=interval, startTime=start_str, endTime=end_str, limit=limit)
        else:
            klines = client.get_historical_klines(symbol, interval, start_str, end_str, limit=limit)
        if not klines or len(klines) < 1:
            break
        fetched_data.extend(klines)
        start_str = klines[-1][0] + 1
        time.sleep(0.5)
    df = pd.DataFrame(fetched_data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    df.drop('ignore', axis=1, inplace=True)
    return df


def save_to_sql(df, table_name):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    # Prepare SQL for conditional insert with reserved keywords handled
    sql = f"""
    IF NOT EXISTS (SELECT 1 FROM {table_name} WHERE [open_time] = ?)
    BEGIN
        INSERT INTO {table_name} ([open_time], [open], [high], [low], [close], [volume], [close_time], [quote_asset_volume], [number_of_trades], [taker_buy_base_asset_volume], [taker_buy_quote_asset_volume])
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    END
    """
    for _, row in df.iterrows():
        cursor.execute(sql, row['open_time'], row['open_time'], row['open'], row['high'], row['low'], row['close'], row['volume'],
                       row['close_time'], row['quote_asset_volume'], row['number_of_trades'],
                       row['taker_buy_base_asset_volume'], row['taker_buy_quote_asset_volume'])
    conn.commit()
    conn.close()
    print(f"Attempted to insert data into table {table_name}, skipping existing records.")

def get_latest_date(table_name):
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    query = f"SELECT MAX(open_time) FROM {table_name}"
    cursor.execute(query)
    result = cursor.fetchone()[0]  # fetchone() returns a tuple, [0] gets the first item from the tuple
    conn.close()
    return result


def main():
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "USDCUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SHIBUSDT", "DOTUSDT", "LINKUSDT", "WBTCUSDT", "UNIUSDT", "TRXUSDT", "MATICUSDT"]
    interval = Client.KLINE_INTERVAL_1DAY

    for symbol in symbols:
        for is_futures in [False, True]:
            table_exists, table_name = check_table_exists(symbol, is_futures)
            if table_exists:
                latest_date = get_latest_date(table_name)
                if latest_date:
                    start_str = int(latest_date.timestamp() * 1000) + 1  # Add 1 ms to avoid overlap
                else:
                    start_str = int(datetime.strptime("1 Jan, 2015", "%d %b, %Y").timestamp() * 1000)
                start_time_str = datetime.fromtimestamp(start_str / 1000).strftime("%d %b, %Y %H:%M:%S")
                df = fetch_klines(symbol, interval, start_time_str, is_futures=is_futures)
                save_to_sql(df, table_name)
                print(f"Data extraction and insertion complete for {table_name}.")

if __name__ == "__main__":
    main()
