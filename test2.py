import ccxt
import pandas as pd
import time
import plotly.graph_objects as go
import os
import json

# Initialize Binance API
binance = ccxt.binance()


def fetch_recent_trades(symbol='BTC/USDT'):
    try:
        trades = binance.fetch_trades(symbol)
        return trades
    except Exception as e:
        print(f"Error fetching trades: {e}")
        return []


def construct_ohlc(trades, interval=60, prev_close=None):
    ohlc_data = []
    start_time = int(trades[0]['timestamp'] / 1000)
    end_time = start_time + interval

    while trades:
        ohlc = {
            'timestamp': start_time,
            'open': prev_close if prev_close is not None else trades[0]['price'],
            'high': trades[0]['price'],
            'low': trades[0]['price'],
            'close': trades[0]['price'],
            'volume': 0
        }

        for trade in trades:
            if int(trade['timestamp'] / 1000) > end_time:
                break
            ohlc['high'] = max(ohlc['high'], trade['price'])
            ohlc['low'] = min(ohlc['low'], trade['price'])
            ohlc['close'] = trade['price']
            ohlc['volume'] += trade['amount']

        prev_close = ohlc['close']
        ohlc_data.append(ohlc)
        start_time = end_time
        end_time += interval
        trades = [t for t in trades if int(t['timestamp'] / 1000) > end_time]

    return ohlc_data, prev_close


# Create a DataFrame to store the data
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(columns=columns)

# Create a Plotly Figure for live updates
fig = go.Figure()

# Set up the initial chart layout
fig.update_layout(
    title='BTC/USDT Candlestick Chart (Past 30 Minutes)',
    yaxis_title='Price (USD)',
    xaxis_title='Time',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Define the path where the file will be saved
output_dir = r'C:\omair\Projects\CapStone Project\Crypto Project results'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
html_file = os.path.join(output_dir, 'btc_usdt_candlestick_chart.html')

# Save the initial empty HTML file
fig.write_html(html_file)


def update_chart(df):
    # Save data as JSON for JavaScript to fetch
    data_file = os.path.join(output_dir, 'data.json')
    df.to_json(data_file, orient='records', date_format='epoch', date_unit='s')

    fig.data = []  # Clear any previous data

    fig.add_trace(go.Candlestick(
        x=pd.to_datetime(df['timestamp'], unit='s'),
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing=dict(line=dict(color='green')),
        decreasing=dict(line=dict(color='red')),
        name='Candlestick'
    ))

    end_time = pd.to_datetime(df['timestamp'].max(), unit='s')
    start_time = end_time - pd.Timedelta(minutes=30)  # Adjusted to 30 minutes

    # Determine y-axis range based on min and max values
    y_min = df['low'].min()
    y_max = df['high'].max()

    fig.update_xaxes(range=[start_time, end_time])
    fig.update_yaxes(range=[y_min, y_max])

    # Save the figure to HTML with JavaScript for live updates
    with open(html_file, 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("""
        <script>
        async function fetchData() {
            const response = await fetch('data.json');
            const data = await response.json();
            const timestamps = data.map(d => new Date(d.timestamp * 1000));
            const open = data.map(d => d.open);
            const high = data.map(d => d.high);
            const low = data.map(d => d.low);
            const close = data.map(d => d.close);

            const update = {
                x: [timestamps],
                open: [open],
                high: [high],
                low: [low],
                close: [close]
            };
            Plotly.update('plotly-graph', update);
        }

        setInterval(fetchData, 60000);  // Fetch new data every 60 seconds
        </script>
        """)


# Main loop to fetch data and update the chart every minute
previous_close = None
while True:
    trades = fetch_recent_trades()
    if trades:
        ohlc_list, previous_close = construct_ohlc(trades, interval=60, prev_close=previous_close)
        new_rows = pd.DataFrame(ohlc_list, columns=columns)
        df = pd.concat([df, new_rows], ignore_index=True)

        # Keep only the most recent 30 minutes of data
        current_time = time.time()
        df = df[df['timestamp'] > (current_time - 1800)]  # Adjusted to 30 minutes

        print(df.tail(1))  # Print the latest entry

        if len(df) > 1:  # Update the chart after at least two entries
            update_chart(df)

    time.sleep(60)  # Wait for 1 minute before fetching new data
