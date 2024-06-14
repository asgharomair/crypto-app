import pyodbc
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, \
    explained_variance_score
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1
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
        print(
            f" - val_mae: {val_mae:.4f} - val_mape: {val_mape:.4f} - val_rmse: {val_rmse:.4f} - val_explained_var: {val_explained_var:.4f}")


# Connection parameters (use environment variables for sensitive data)
server = os.getenv('DB_SERVER', 'localhost')
database = os.getenv('DB_NAME', 'CryptoPredictor')
username = os.getenv('DB_USER', 'sa')
password = os.getenv('DB_PASSWORD', 'Pakistan@12')


# Function to create database connection
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


# Function to load data from the database
def load_data(features, target_variable):
    conn = create_connection()
    if conn is not None:
        try:
            sql_query = 'SELECT {} FROM Data'.format(
                ', '.join([f'[{feature}]' for feature in features + [target_variable]]))
            data = pd.read_sql(sql_query, conn)
            conn.close()
            return data
        except Exception as e:
            print("Error loading data: ", e)
            return pd.DataFrame()
    else:
        return pd.DataFrame()


# Function to preprocess data
def preprocess_data(data, features, target_variable, scaler_path):
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).view('int64') / 10 ** 9

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=features + [target_variable], inplace=True)

    numeric_features = data[features].select_dtypes(include=[np.number])
    y = data[target_variable].astype(float)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(numeric_features)

    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved at {scaler_path}")

    return X_scaled, y


# Function to build the model
def build_complex_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation='relu', kernel_regularizer=l1(0.001))(input_layer)
    x = Dropout(0.4)(x)
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


# Function to evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print("Train MAE:", train_mae)
    print("Test MAE:", test_mae)
    print("Train R^2:", train_r2)
    print("Test R^2:", test_r2)


# Function to train the model
def train_model(target_variable, sheet_name):
    feature_file = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\Feature_Selection_Results.xlsx'
    features = pd.read_excel(feature_file, sheet_name=sheet_name).iloc[:, 0].tolist()
    data = load_data(features, target_variable)
    if not data.empty:
        output_folder = 'C:\\omair\\Projects\\CapStone Project\\Crypto Project results\\BTC Model'
        scaler_path = os.path.join(output_folder, f'{target_variable}_RobustScaler_{sheet_name}.pkl')
        X_scaled, y = preprocess_data(data, features, target_variable, scaler_path)
        if X_scaled is not None:
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
            model = build_complex_model(X_train.shape[1])
            additional_metrics = AdditionalMetrics(validation_data=(X_val, y_val))
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=0.0001,
                                          min_lr=1e-6)
            history = model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=1, validation_data=(X_val, y_val),
                                callbacks=[additional_metrics, reduce_lr])
            evaluate_model(model, X_train, y_train, X_val, y_val)

            # Save the plots instead of displaying them
            plt.figure(figsize=(14, 7))
            plt.subplot(2, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.subplot(2, 2, 2)
            plt.plot(history.epoch, history.history['val_rmse'], label='Validation RMSE')
            plt.title('Model RMSE')
            plt.ylabel('RMSE')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.subplot(2, 2, 3)
            plt.plot(history.epoch, history.history['val_mape'], label='Validation MAPE')
            plt.title('Model MAPE')
            plt.ylabel('MAPE')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.subplot(2, 2, 4)
            plt.plot(history.epoch, history.history['val_explained_var'], label='Validation Explained Variance')
            plt.title('Explained Variance')
            plt.ylabel('Explained Variance')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.suptitle(f'Dense Model - {target_variable} - {sheet_name}', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save the plot
            plot_save_path = os.path.join(output_folder, f'{target_variable}_Training_Plot_{sheet_name}.png')
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Plot saved at {plot_save_path}")

            # Save the model
            model_save_path = os.path.join(output_folder, f'{target_variable}_Dense_Model_{sheet_name}.h5')
            model.save(model_save_path)
            print(f"Model saved at {model_save_path}")
        else:
            print("Error in data preprocessing.")
    else:
        print("No data loaded to process.")


# Train models for each target variable and feature selection method
target_features_map = {
    'target_BTC_Price': ['target_BTC_Pric_Correlation', 'target_BTC_Pric_Granger', 'target_BTC_Pric_RandomForest',
                         'target_BTC_Pric_SHAP'],
    'Target_XRP_Price': ['Target_XRP_Pric_Correlation', 'Target_XRP_Pric_Granger', 'Target_XRP_Pric_RandomForest',
                         'Target_XRP_Pric_SHAP'],
    'Target_SOL_Price': ['Target_SOL_Pric_Correlation', 'Target_SOL_Pric_Granger', 'Target_SOL_Pric_RandomForest',
                         'Target_SOL_Pric_SHAP'],
    'Target_ETH_Price': ['Target_ETH_Pric_Correlation', 'Target_ETH_Pric_Granger', 'Target_ETH_Pric_RandomForest',
                         'Target_ETH_Pric_SHAP'],
    'Target_BNB_Price': ['Target_BNB_Pric_Correlation', 'Target_BNB_Pric_Granger', 'Target_BNB_Pric_RandomForest',
                         'Target_BNB_Pric_SHAP']
}

for target_variable, feature_sheets in target_features_map.items():
    for sheet_name in feature_sheets:
        train_model(target_variable, sheet_name)
