import numpy as np
import pandas as pd
import pyodbc
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, explained_variance_score
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, Input, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import os



# Callback class for additional metrics
class AdditionalMetrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_x, val_y = self.validation_data
        val_pred = self.model.predict(val_x)
        val_mae = mean_absolute_error(val_y, val_pred)
        val_mape = mean_absolute_percentage_error(val_y, val_pred)
        val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
        val_explained_var = explained_variance_score(val_y, val_pred)
        logs['val_mae'] = val_mae
        logs['val_mape'] = val_mape
        logs['val_rmse'] = val_rmse
        logs['val_explained_var'] = val_explained_var
        print(f"Epoch {epoch + 1} - val_mae: {val_mae:.4f} - val_mape: {val_mape:.4f} - val_rmse: {val_rmse:.4f} - val_explained_var: {val_explained_var:.4f}")


def create_connection():
    server = os.getenv('DB_SERVER', 'localhost')
    database = os.getenv('DB_NAME', 'CryptoPredictor')
    username = os.getenv('DB_USER', 'sa')
    password = os.getenv('DB_PASSWORD', 'Pakistan@12')
    try:
        conn = pyodbc.connect(
            r'DRIVER={ODBC Driver 17 for SQL Server};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )
        return conn
    except pyodbc.Error as e:
        print("Error in connection: ", e)
        return None
selected_features = [
    'close_BTCUSDT_historical_data_spot',
    'high_BTCUSDT_historical_data_spot',
    'low_BTCUSDT_historical_data_spot',
    'BTCUSDT_EMA5',
    'open_BTCUSDT_historical_data_spot',
    'close_WBTCUSDT_historical_data_spot',
    'close_BTCUSDT_historical_data_futures',
    'high_WBTCUSDT_historical_data_spot',
    'low_WBTCUSDT_historical_data_spot',
    'high_BTCUSDT_historical_data_futures',
    'BTCUSDT_EMA10',
    'low_BTCUSDT_historical_data_futures',
    'open_WBTCUSDT_historical_data_spot',
    'open_BTCUSDT_historical_data_futures',
    'Target_ETH_Price',
    'close_ETHUSDT_historical_data_spot',
    'high_ETHUSDT_historical_data_spot',
    'low_ETHUSDT_historical_data_spot',
    'open_ETHUSDT_historical_data_spot',
    'close_ETHUSDT_historical_data_futures',
    'low_ETHUSDT_historical_data_futures',
    'high_ETHUSDT_historical_data_futures',
    'High_NASDAQ',
    'Open_NASDAQ',
    'Low_NASDAQ',
    '[Close/Last_NASDAQ]',
    'open_ETHUSDT_historical_data_futures',
    'Target_BNB_Price',
    'high_BNBUSDT_historical_data_spot',
    'close_BNBUSDT_historical_data_spot',
    'High_NYSE',
    'close_time_WBTCUSDT_historical_data_spot',
    'open_BNBUSDT_historical_data_spot',
    'Close_NYSE',
    'low_BNBUSDT_historical_data_spot',
    'Open_NYSE',
    'Low_NYSE',
    'close_SOLUSDT_historical_data_futures',
    'low_SOLUSDT_historical_data_futures',
    'high_SOLUSDT_historical_data_futures',
    'open_SOLUSDT_historical_data_futures',
    'high_TRXUSDT_historical_data_spot',
    'close_TRXUSDT_historical_data_spot',
    'open_TRXUSDT_historical_data_spot',
    'low_TRXUSDT_historical_data_spot',
    'close_time_USDCUSDT_historical_data_futures',
    'low_LINKUSDT_historical_data_spot',
    'close_LINKUSDT_historical_data_spot',
    'open_LINKUSDT_historical_data_spot',
    'high_LINKUSDT_historical_data_spot',
    'low_LINKUSDT_historical_data_futures',
    'close_LINKUSDT_historical_data_futures',
    'close_BNBUSDT_historical_data_futures',
    'open_LINKUSDT_historical_data_futures',
    'high_BNBUSDT_historical_data_futures',
    'high_LINKUSDT_historical_data_futures',
    'low_BNBUSDT_historical_data_futures',
    'open_BNBUSDT_historical_data_futures',
    'high_TRXUSDT_historical_data_futures',
    'close_TRXUSDT_historical_data_futures',
    'open_TRXUSDT_historical_data_futures',
    'low_ADAUSDT_historical_data_spot',
    'close_ADAUSDT_historical_data_spot',
    'low_TRXUSDT_historical_data_futures',
    'open_ADAUSDT_historical_data_spot',
    'high_ADAUSDT_historical_data_spot',
    'Target_SOL_Price',
    'close_SOLUSDT_historical_data_spot',
    'high_SOLUSDT_historical_data_spot',
    'low_SOLUSDT_historical_data_spot',
    'open_SOLUSDT_historical_data_spot',
    'Target_XRP_Price',
    'low_XRPUSDT_historical_data_spot',
    'close_XRPUSDT_historical_data_spot',
    'open_XRPUSDT_historical_data_spot',
    'close_DOGEUSDT_historical_data_futures',
    'high_XRPUSDT_historical_data_spot',
    'high_DOGEUSDT_historical_data_futures',
    'low_DOGEUSDT_historical_data_spot',
    'open_DOGEUSDT_historical_data_futures',
    'low_DOGEUSDT_historical_data_futures',
    'Low_GoldFutures',
    'Price_GoldFutures',
    'Open_GoldFutures',
    'High_GoldFutures',
    'close_DOGEUSDT_historical_data_spot',
    'open_DOGEUSDT_historical_data_spot',
    'price_variation_BTCUSDT_historical_data_spot',
    'number_of_trades_WBTCUSDT_historical_data_spot',
    'close_XRPUSDT_historical_data_futures',
    'low_XRPUSDT_historical_data_futures',
    'open_XRPUSDT_historical_data_futures',
    'Year',
    'high_DOGEUSDT_historical_data_spot',
    'high_XRPUSDT_historical_data_futures',
    'Date',
    'close_time_BNBUSDT_historical_data_spot',
    'close_time_ETHUSDT_historical_data_spot',
    'close_time_BTCUSDT_historical_data_spot',
    'close_time_USDCUSDT_historical_data_spot',
    'low_DOTUSDT_historical_data_spot',
    'close_time_ADAUSDT_historical_data_spot',
    'close_DOTUSDT_historical_data_spot',
    'close_time_XRPUSDT_historical_data_spot',
    'number_of_trades_ETHUSDT_historical_data_spot',
    'open_DOTUSDT_historical_data_spot',
    'high_DOTUSDT_historical_data_spot',
    'close_time_TRXUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_ETHUSDT_historical_data_spot',
    'price_variation_SOLUSDT_historical_data_futures',
    'quote_asset_volume_ETHUSDT_historical_data_spot',
    'price_variation_WBTCUSDT_historical_data_spot',
    'close_MATICUSDT_historical_data_spot',
    'price_variation_BNBUSDT_historical_data_spot',
    'low_UNIUSDT_historical_data_spot',
    'high_MATICUSDT_historical_data_spot',
    'low_MATICUSDT_historical_data_spot',
    'open_MATICUSDT_historical_data_spot',
    'close_UNIUSDT_historical_data_spot',
    'open_UNIUSDT_historical_data_spot',
    'price_variation_ETHUSDT_historical_data_spot',
    'close_AVAXUSDT_historical_data_futures',
    'high_UNIUSDT_historical_data_spot',
    'high_AVAXUSDT_historical_data_futures',
    'close_SHIBUSDT_historical_data_spot',
    'low_AVAXUSDT_historical_data_futures',
    'number_of_trades_BNBUSDT_historical_data_spot',
    'open_AVAXUSDT_historical_data_futures',
    'high_SHIBUSDT_historical_data_spot',
    'open_SHIBUSDT_historical_data_spot',
    'low_SHIBUSDT_historical_data_spot',
    'high_AVAXUSDT_historical_data_spot',
    'close_AVAXUSDT_historical_data_spot',
    'close_UNIUSDT_historical_data_futures',
    'open_AVAXUSDT_historical_data_spot',
    'open_UNIUSDT_historical_data_futures',
    'high_UNIUSDT_historical_data_futures',
    'close_time_LINKUSDT_historical_data_spot',
    'low_UNIUSDT_historical_data_futures',
    'low_AVAXUSDT_historical_data_spot',
    'close_DOTUSDT_historical_data_futures',
    'low_DOTUSDT_historical_data_futures',
    'high_DOTUSDT_historical_data_futures',
    'open_DOTUSDT_historical_data_futures',
    'price_variation_SOLUSDT_historical_data_spot',
    'price_variation_BTCUSDT_historical_data_futures',
    'quote_asset_volume_BNBUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_BNBUSDT_historical_data_spot',
    'close_ADAUSDT_historical_data_futures',
    'low_ADAUSDT_historical_data_futures',
    'open_ADAUSDT_historical_data_futures',
    'high_ADAUSDT_historical_data_futures',
    'taker_buy_base_asset_volume_USDCUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_USDCUSDT_historical_data_spot',
    'close_time_MATICUSDT_historical_data_spot',
    'number_of_trades_SHIBUSDT_historical_data_spot',
    'volume_USDCUSDT_historical_data_spot',
    'quote_asset_volume_USDCUSDT_historical_data_spot',
    'price_variation_ADAUSDT_historical_data_spot',
    'quote_asset_volume_SOLUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_SOLUSDT_historical_data_spot',
    'price_variation_AVAXUSDT_historical_data_spot',
    'price_variation_AVAXUSDT_historical_data_futures',
    'number_of_trades_LINKUSDT_historical_data_spot',
    'number_of_trades_TRXUSDT_historical_data_spot',
    'number_of_trades_ADAUSDT_historical_data_spot',
    'close_time_DOGEUSDT_historical_data_spot',
    'number_of_trades_USDCUSDT_historical_data_spot',
    'quote_asset_volume_WBTCUSDT_historical_data_spot',
    'price_variation_LINKUSDT_historical_data_futures',
    'number_of_trades_SOLUSDT_historical_data_spot',
    'number_of_trades_XRPUSDT_historical_data_spot',
    'price_variation_ETHUSDT_historical_data_futures',
    'quote_asset_volume_TRXUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_TRXUSDT_historical_data_spot',
    'price_variation_LINKUSDT_historical_data_spot',
    'quote_asset_volume_SOLUSDT_historical_data_futures',
    'quote_asset_volume_XRPUSDT_historical_data_spot',
    'taker_buy_quote_asset_volume_SOLUSDT_historical_data_futures',
    'taker_buy_quote_asset_volume_XRPUSDT_historical_data_spot'

]  # Same as before

def load_data():
    conn = create_connection()
    if conn is not None:
        try:

            sql_query = 'SELECT {} FROM Data'.format(
                ', '.join(selected_features + ['target_BTC_Price']))
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def preprocess_data(data, sequence_length=10):
    # Exclude 'Date' or convert it to numeric
    if 'Date' in data.columns:
        # To exclude 'Date' column
        # data = data.drop(columns=['Date'])
        # Or convert 'Date' to numeric timestamp
        data['Date'] = pd.to_datetime(data['Date']).view('int64') / 10**9


    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    scaler = RobustScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    y = []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i - sequence_length:i])
        y.append(data_scaled[i, -1])  # Assuming the last column is the target variable
    return np.array(X), np.array(y)



def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)


def main():
    data = load_data()
    if not data.empty:
        X, y = preprocess_data(data)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

        # Define Transformer parameters
        head_size = 256
        num_heads = 4
        ff_dim = 4
        num_transformer_blocks = 4
        mlp_units = [128]
        dropout = 0.1
        mlp_dropout = 0.1

        # Build the model
        model = build_transformer_model(
            X_train.shape[1:],
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            dropout=dropout,
            mlp_dropout=mlp_dropout
        )

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Initialize the custom callback
        additional_metrics = AdditionalMetrics(validation_data=(X_val, y_val))

        # Train the model
        history = model.fit(
            X_train, y_train, epochs=500, batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[additional_metrics]
        )

        # Evaluate and plot the results
        evaluate_model(model, X_train, y_train, X_val, y_val)

        # Save the model
        model_save_path = os.path.join('C:\\omair\\Projects\\CapStone Project\\Crypto Project results',
                                       'BTC_Transformer.keras')
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
        model.save(model_save_path)
        print(f"Model saved at {model_save_path}")


        # Plot metrics
        plt.figure(figsize=(14, 7))

        # Plot Training & Validation Loss
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        # Plot RMSE
        plt.subplot(2, 2, 2)
        plt.plot(history.epoch, history.history['val_rmse'], label='Validation RMSE')
        plt.title('Model RMSE')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        # Plot MAPE
        plt.subplot(2, 2, 3)
        plt.plot(history.epoch, history.history['val_mape'], label='Validation MAPE')
        plt.title('Model MAPE')
        plt.ylabel('MAPE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        # Plot Explained Variance
        plt.subplot(2, 2, 4)
        plt.plot(history.epoch, history.history['val_explained_var'], label='Validation Explained Variance')
        plt.title('Explained Variance')
        plt.ylabel('Explained Variance')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')

        plt.suptitle('Transformers Model', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def evaluate_model(model, X_train, y_train, X_val, y_val):
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)  # Use the correct variable here instead of X_test
    train_mse = mean_squared_error(y_train, train_pred)
    val_mse = mean_squared_error(y_val, val_pred)  # Use the correct variable here instead of y_test
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)  # Use the correct variable here instead of y_test
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)  # Use the correct variable here instead of y_test

    print(f"Train MSE: {train_mse:.4f}")
    print(f"Val MSE: {val_mse:.4f}")  # Use the correct terminology for consistency
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Val MAE: {val_mae:.4f}")  # Use the correct terminology for consistency
    print(f"Train R²: {train_r2:.4f}")
    print(f"Val R²: {val_r2:.4f}")  # Use the correct terminology for consistency


if __name__ == "__main__":
    main()