import pyodbc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2, l1
import os
import matplotlib.pyplot as plt
import joblib


# Define the callback class to compute additional metrics after each epoch
class AdditionalMetrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_x, val_y = self.validation_data
        val_pred = self.model.predict(val_x)

        # Compute additional metrics
        val_mae = mean_absolute_error(val_y, val_pred)
        val_mape = mean_absolute_percentage_error(val_y, val_pred)
        val_rmse = np.sqrt(mean_squared_error(val_y, val_pred))
        val_explained_var = explained_variance_score(val_y, val_pred)

        # Log the additional metrics
        logs['val_mae'] = val_mae
        logs['val_mape'] = val_mape
        logs['val_rmse'] = val_rmse
        logs['val_explained_var'] = val_explained_var
        print(f" - val_mae: {val_mae:.4f} - val_mape: {val_mape:.4f} - val_rmse: {val_rmse:.4f} - val_explained_var: {val_explained_var:.4f}")

# Connection parameters (use environment variables for sensitive data)
server = os.getenv('DB_SERVER', 'localhost')
database = os.getenv('DB_NAME', 'CryptoPredictor')
username = os.getenv('DB_USER', 'sa')
password = os.getenv('DB_PASSWORD', 'Pakistan@12')

# Selected features based on correlation analysis
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
    'Close/Last_NASDAQ',
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
    'taker_buy_quote_asset_volume_XRPUSDT_historical_data_spot',

    # 'Date',  # Date removed to simplify the data handling
]

def create_connection():
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

def load_data():
    conn = create_connection()
    if conn is not None:
        try:
            # Wrap column names in square brackets to avoid keyword conflicts
            sql_query = 'SELECT {} FROM Data'.format(', '.join([f'[{feature}]' for feature in selected_features + ['target_BTC_Price']]))
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def preprocess_data(data):
    # Replace infinities with NaN, then drop rows with NaN values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=selected_features + ['target_BTC_Price'], inplace=True)

    # Select only the numeric features for scaling (excluding the target variable)
    numeric_features = data[selected_features].select_dtypes(include=[np.number])

    # Ensure that the target variable is included in the DataFrame and is numeric
    target_column = 'target_BTC_Price'
    y = data[target_column].astype(float)  # Convert target to float to ensure it's numeric

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(numeric_features)

    # Save the scaler
    scaler_path = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\robust_scaler.pkl'
    joblib.dump(scaler, scaler_path)

    return X_scaled, y

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    # Print evaluation metrics
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("Train MAE:", train_mae)
    print("Test MAE:", test_mae)
    print("Train R^2:", train_r2)
    print("Test R^2:", test_r2)

def build_complex_model(input_dim):
    # Define the model architecture
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation='relu', kernel_regularizer=l1(0.001))(input_layer)
    x = Dropout(0.4)(x)  # Increased dropout rate
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def main():
    data = load_data()
    if not data.empty:
        X_scaled, y = preprocess_data(data)
        if X_scaled is not None:
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

            # Build the model
            model = build_complex_model(X_train.shape[1])

            # Instantiate the AdditionalMetrics callback
            additional_metrics = AdditionalMetrics(validation_data=(X_val, y_val))

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                min_delta=0.0001, cooldown=0, min_lr=1^-6)

            # Train the model using the .fit() method
            history = model.fit(
                X_train, y_train,
                epochs=500,
                batch_size=32,
                verbose=1,
                validation_data=(X_val, y_val),
                callbacks=[additional_metrics, reduce_lr]
            )

            # Evaluate the model's performance
            evaluate_model(model, X_train, y_train, X_val, y_val)

            print(f"Shape of X_train: {X_train.shape}")
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

            plt.suptitle('Dense Model', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

            model_save_path = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\BTC_Dense_Model.h5'
            model.save(model_save_path)
            print(f"Model saved at {model_save_path}")
        else:
            print("Error in data preprocessing.")
    else:
        print("No data loaded to process.")

if __name__ == "__main__":
    main()

print(f"Number of selected features: {len(selected_features)}")
