import requests
import time
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from collections import deque
import plotly.graph_objs as go
import os  # Import os to get the PORT from environment variables


def fetch_bitcoin_price():
    url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['bitcoin']['usd']
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None


# Initialize the Dash app
app = dash.Dash(__name__)

# Define a deque to store the data
X = deque(maxlen=60)  # 60 points for 1 hour at 1-minute intervals
Y = deque(maxlen=60)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Live Bitcoin Price Tracker", style={'textAlign': 'center', 'color': '#007bff'}),
    dcc.Graph(id='live-graph', animate=True, style={'height': '60vh'}),
    dcc.Interval(
        id='graph-update',
        interval=60 * 1000,  # in milliseconds (60 seconds)
        n_intervals=0
    )
], style={'padding': '20px', 'fontFamily': 'Arial'})


@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(n):
    current_time = datetime.now()
    current_price = fetch_bitcoin_price()

    if current_price is not None:
        X.append(current_time)
        Y.append(current_price)

    x_range = [datetime.now() - timedelta(hours=1), datetime.now()]

    data = go.Scatter(
        x=list(X),
        y=list(Y),
        mode='lines',
        line=dict(color='blue', width=2),
        name='Price (USD)'
    )

    layout = go.Layout(
        xaxis=dict(range=x_range, tickformat='%H:%M', title='Time (Last 1 Hour)',
                   titlefont=dict(size=14, color='black')),
        yaxis=dict(range=[min(Y) - 50, max(Y) + 50] if Y else [0, 1], title='Price (USD)',
                   titlefont=dict(size=14, color='black')),
        title='Bitcoin Price (USD)',
        titlefont=dict(size=20, color='black'),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#ffffff',
        margin={'l': 50, 'r': 10, 't': 50, 'b': 40},
        hovermode='closest',
        showlegend=False
    )

    return {'data': [data], 'layout': layout}


# Run the Dash app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use the PORT environment variable or default to 5000
    app.run_server(host='0.0.0.0', port=port, debug=False, use_reloader=False)
