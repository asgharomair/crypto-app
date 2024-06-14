import os
import pyodbc
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests
import shap
import xgboost
import openpyxl
import numpy as np

# Read column headers from file
data_table_path = r'C:\omair\Projects\CapStone Project\Crypto Project results\Data_Table.csv'
columns_df = pd.read_csv(data_table_path)
columns = columns_df.columns.tolist()

# Database credentials
server = 'localhost'
database = 'CryptoPredictor'
username = 'sa'
password = 'Pakistan@12'
driver = '{ODBC Driver 17 for SQL Server}'

# Connect to the database and load data
conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
query = "SELECT * FROM Data"
df = pd.read_sql(query, conn)
conn.close()

# List of target columns
target_columns = ['target_BTC_Price', 'Target_XRP_Price', 'Target_SOL_Price', 'Target_ETH_Price', 'Target_BNB_Price']

# Initialize Excel writer
output_path = r'C:\omair\Projects\CapStone Project\Crypto Project results\Feature_Selection_Results.xlsx'
writer = pd.ExcelWriter(output_path, engine='openpyxl')

# Sheet colors (one color per target)
colors = ['00FFCC99', '00CCFFCC', '00FFCCFF', '00CCCCFF', '00FFFF99']

for target_column, color in zip(target_columns, colors):
    # Drop rows with missing target values and separate features and target
    df.dropna(subset=[target_column], inplace=True)
    X = df.drop(columns=['Date', target_column])
    y = df[target_column]

    # Drop columns with all missing values
    X = X.dropna(axis=1, how='all')

    # Exclude columns that start with "target" or "Target"
    X = X[[col for col in X.columns if not col.lower().startswith('target')]]

    # Impute missing values in features
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Convert imputed data back to DataFrame
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

    # Replace infinite values with NaNs and then impute them
    X_imputed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_imputed), columns=X.columns)

    # Initialize results dictionary for the current target
    results = {}

    # 1. Correlation with the target column
    correlation = df.corr()[target_column].drop(target_column)
    selected_correlation_features = correlation[correlation.abs() > 0.1].index.tolist()
    results['Correlation'] = selected_correlation_features

    # 2. Granger Causality Test
    max_lag = 10
    granger_features = {}
    for col in X.columns:
        try:
            test_result = grangercausalitytests(df[[target_column, col]].dropna(), max_lag)
            p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]
            granger_features[col] = min(p_values)  # Keep the minimum p-value for each feature
        except:
            granger_features[col] = 1.0  # Assign a high p-value if the test fails

    selected_granger_features = [feature for feature, p_value in granger_features.items() if p_value < 0.05]
    results['Granger'] = selected_granger_features

    # 3. Feature Importance with Random Forest
    rf_model = RandomForestRegressor()
    rf_model.fit(X_imputed, y)
    importances = rf_model.feature_importances_
    selected_rf_features = [feature for feature, importance in zip(X.columns, importances) if importance > 0.01]
    results['RandomForest'] = selected_rf_features

    # 4. SHAP Values
    xgb_model = xgboost.XGBRegressor()
    xgb_model.fit(X_imputed, y)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_imputed)
    shap_importance = pd.DataFrame(list(zip(X.columns, np.abs(shap_values).mean(axis=0))), columns=['Feature', 'SHAP Importance'])
    shap_importance.sort_values(by='SHAP Importance', ascending=False, inplace=True)
    selected_shap_features = shap_importance['Feature'].tolist()[:10]  # Select top 10 features
    results['SHAP'] = selected_shap_features

    # Save results to Excel
    for method, features in results.items():
        df_result = pd.DataFrame(features, columns=['Selected Features'])
        sheet_name = f"{target_column[:15]}_{method}"  # Shorten sheet name to 31 characters max
        df_result.to_excel(writer, sheet_name=sheet_name, index=False)

    # Set tab color
    workbook = writer.book
    for sheetname in writer.sheets:
        if sheetname.startswith(target_column[:15]):
            workbook[sheetname].sheet_properties.tabColor = color

# Save the Excel file
writer.close()
print("Feature selection results saved to Excel file with colored tabs.")
