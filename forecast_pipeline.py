"""
========================================================================================
Supermarket Sales Forecasting Machine Learning Pipeline
Target Database: SupermarketSalesDB (Microsoft SQL Server)
Source View    : dbo.MonthlySales
Models         : Linear Regression vs Random Forest Regressor
Evaluation     : RMSE, MAE, MAPE, R-squared
Output         : Multi-step future sales forecast (INR ₹)
========================================================================================
"""

import sys
import pyodbc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DATABASE CONNECTION CONFIGURATION
# ==============================================================================
# Connection options:
# Option A: Windows Authentication (Most common for local SSMS)
# Option B: SQL Server Authentication (Username & Password)

DB_CONFIG = {
    "server": r"localhostSQLEXPRESS",       # Default for SQL Server Express. Try "localhost" for Developer Edition.
    "database": "SupermarketSalesDB",
    "driver": "{ODBC Driver 17 for SQL Server}", # or "{ODBC Driver 18 for SQL Server}"
    "trusted_connection": "yes",          # Set to 'yes' for Windows Auth
    # "uid": "sa",                         # Uncomment if using SQL Auth
    # "pwd": "YourStrongPassword123!",     # Uncomment if using SQL Auth
}

def get_connection():
    """
    Builds and returns a pyodbc connection to SupermarketSalesDB.
    Automatically tests common instance names (localhost\SQLEXPRESS, localhost, .\SQLEXPRESS, 127.0.0.1,1433)
    to handle Error 08001 seamlessly.
    """
    # Candidate server instance names to probe
    configured_srv = DB_CONFIG.get("server", r"localhostSQLEXPRESS")
    candidates = [
        configured_srv,
        r"localhostSQLEXPRESS",
        r".SQLEXPRESS",
        "localhost",
        "127.0.0.1,1433",
        r"(localdb)mssqllocaldb"
    ]
    # Remove duplicates preserving order
    seen = set()
    candidate_list = [c for c in candidates if not (c in seen or seen.add(c))]

    drivers_to_try = [
        DB_CONFIG.get("driver", "{ODBC Driver 17 for SQL Server}"),
        "{ODBC Driver 18 for SQL Server}",
        "{ODBC Driver 17 for SQL Server}",
        "{SQL Server}"
    ]
    seen_drv = set()
    driver_list = [d for d in drivers_to_try if not (d in seen_drv or seen_drv.add(d))]

    last_error = None
    for srv in candidate_list:
        for drv in driver_list:
            try:
                if DB_CONFIG.get("trusted_connection") == "yes":
                    conn_str = (
                        f"DRIVER={drv};"
                        f"SERVER={srv};"
                        f"DATABASE={DB_CONFIG['database']};"
                        f"Trusted_Connection=yes;"
                        f"TrustServerCertificate=yes;"
                    )
                else:
                    conn_str = (
                        f"DRIVER={drv};"
                        f"SERVER={srv};"
                        f"DATABASE={DB_CONFIG['database']};"
                        f"UID={DB_CONFIG['uid']};"
                        f"PWD={DB_CONFIG['pwd']};"
                        f"TrustServerCertificate=yes;"
                    )
                conn = pyodbc.connect(conn_str, timeout=3)
                print(f"Connected successfully to SupermarketSalesDB on '{srv}' using {drv}!")
                return conn
            except Exception as e:
                last_error = e

    print("\n[ERROR 08001] Could not connect to SQL Server on any tested instance:")
    for c in candidate_list:
        print(f"  - Tried: {c}")
    print(f"Detail: {last_error}")
    print("\nHow to fix:")
    print("  1. Check the exact server name in your SSMS login screen (e.g. DESKTOP-XYZ\SQLEXPRESS).")
    print("  2. In SQL Server Configuration Manager -> SQL Server Network Configuration -> Protocols -> Enable TCP/IP.")
    print("  3. Ensure SQL Server Browser service is running for named instances.")
    raise last_error

# ==============================================================================
# 2. DATA EXTRACTION FROM MONTHLYSALES VIEW
# ==============================================================================
def load_monthly_sales_data(conn):
    """Fetches chronological monthly sales data from dbo.MonthlySales view."""
    query = """
    SELECT 
        [Year],
        [Month],
        [MonthName],
        CAST(TotalSales AS FLOAT) AS TotalSales,
        TotalQuantity,
        NumberOfTransactions
    FROM dbo.MonthlySales
    ORDER BY [Year] ASC, [Month] ASC;
    """
    df = pd.read_sql(query, conn)
    # Create a proper datetime index (1st of each month)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Loaded {len(df)} monthly records spanning {df['Date'].min().strftime('%b %Y')} to {df['Date'].max().strftime('%b %Y')}.")
    return df

# ==============================================================================
# 3. FEATURE ENGINEERING FOR TIME-SERIES ML FORECASTING
# ==============================================================================
def create_features(df, target_col="TotalSales"):
    """
    Transforms raw time series into supervised ML tabular features:
      1. Trend Index (t = 1, 2, ...)
      2. Lag Features (Sales_lag_1, Sales_lag_2, Sales_lag_3)
      3. Rolling Window Aggregates (3-Month Rolling Mean, Rolling Std)
      4. Seasonal Calendar Features (Month sin/cos encoding, Quarter)
      5. Indian Festive Season Flag (Diwali/Navratri surge in Oct-Nov)
    """
    df_feat = df.copy()
    
    # 1. Linear Time Trend
    df_feat["TimeIndex"] = np.arange(1, len(df_feat) + 1)
    
    # 2. Calendar and Seasonality Features
    df_feat["Quarter"] = df_feat["Date"].dt.quarter
    # Cyclical sin/cos encoding for month (smooth transition Dec -> Jan)
    df_feat["Month_Sin"] = np.sin(2 * np.pi * df_feat["Month"] / 12.0)
    df_feat["Month_Cos"] = np.cos(2 * np.pi * df_feat["Month"] / 12.0)
    
    # Festive Season Flag (India Diwali / Dussehra festival shopping peak: Oct & Nov)
    df_feat["IsFestiveSeason"] = df_feat["Month"].isin([10, 11]).astype(int)
    
    # 3. Lag Features
    df_feat["Sales_Lag_1"] = df_feat[target_col].shift(1)
    df_feat["Sales_Lag_2"] = df_feat[target_col].shift(2)
    df_feat["Sales_Lag_3"] = df_feat[target_col].shift(3)
    
    # 4. Rolling Statistics (3-month window)
    df_feat["Rolling_Mean_3"] = df_feat[target_col].shift(1).rolling(window=3).mean()
    df_feat["Rolling_Std_3"] = df_feat[target_col].shift(1).rolling(window=3).std()
    
    # Drop initial rows with NaN from lags
    df_clean = df_feat.dropna().reset_index(drop=True)
    return df_clean

# ==============================================================================
# 4. MODEL TRAINING & COMPARISON (LINEAR REGRESSION vs RANDOM FOREST)
# ==============================================================================
def train_and_evaluate_models(df_features, test_months=6):
    """
    Splits time series chronologically into Train & Test sets.
    Trains Linear Regression and Random Forest Regressor.
    Returns metrics and trained models.
    """
    feature_cols = [
        "TimeIndex", 
        "Month_Sin", 
        "Month_Cos", 
        "Quarter", 
        "IsFestiveSeason",
        "Sales_Lag_1", 
        "Sales_Lag_2", 
        "Sales_Lag_3", 
        "Rolling_Mean_3"
    ]
    target_col = "TotalSales"
    
    # Chronological Split (Preserves time dependency!)
    split_idx = len(df_features) - test_months
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    print(f"\n--- Training Partition: {len(X_train)} months | Test Partition: {len(X_test)} months ---")
    
    # 4.1 Linear Regression Model (Captures Global Trend & Baseline Seasonality)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    # 4.2 Random Forest Regressor (Captures Non-linear dynamics & interactions)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    
    # Helper to calculate metrics
    def calculate_metrics(y_true, y_pred, model_name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        return {
            "Model": model_name,
            "RMSE (INR ₹)": round(rmse, 2),
            "MAE (INR ₹)": round(mae, 2),
            "MAPE (%)": round(mape, 2),
            "R² Score": round(r2, 4)
        }
    
    metrics_lr = calculate_metrics(y_test, lr_preds, "Linear Regression")
    metrics_rf = calculate_metrics(y_test, rf_preds, "Random Forest")
    
    comparison_df = pd.DataFrame([metrics_lr, metrics_rf])
    print("\n=== MODEL EVALUATION RESULTS ON HOLD-OUT TEST SET ===")
    print(comparison_df.to_string(index=False))
    
    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "lr_preds": lr_preds,
        "rf_preds": rf_preds,
        "comparison_df": comparison_df
    }

# ==============================================================================
# 5. RECURSIVE FUTURE SALES FORECASTING (2027 PROJECTIONS)
# ==============================================================================
def forecast_future_sales(model, df_features, feature_cols, horizon_months=6):
    """
    Autoregressive iterative forecasting into future months (e.g. 2027).
    Feeds predicted sales as lags for subsequent periods.
    """
    last_row = df_features.iloc[-1]
    last_date = last_row["Date"]
    last_t = last_row["TimeIndex"]
    
    # Historical recent buffer for lags
    sales_history = list(df_features["TotalSales"].values)
    
    future_records = []
    current_date = last_date
    current_t = last_t
    
    for i in range(1, horizon_months + 1):
        # Move forward by 1 month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
        current_t += 1
        
        m = current_date.month
        q = (m - 1) // 3 + 1
        m_sin = np.sin(2 * np.pi * m / 12.0)
        m_cos = np.cos(2 * np.pi * m / 12.0)
        is_festive = 1 if m in [10, 11] else 0
        
        lag_1 = sales_history[-1]
        lag_2 = sales_history[-2]
        lag_3 = sales_history[-3]
        roll_3 = np.mean([lag_1, lag_2, lag_3])
        
        feat_vector = pd.DataFrame([{
            "TimeIndex": current_t,
            "Month_Sin": m_sin,
            "Month_Cos": m_cos,
            "Quarter": q,
            "IsFestiveSeason": is_festive,
            "Sales_Lag_1": lag_1,
            "Sales_Lag_2": lag_2,
            "Sales_Lag_3": lag_3,
            "Rolling_Mean_3": roll_3
        }])[feature_cols]
        
        predicted_val = model.predict(feat_vector)[0]
        # Ensure sales cannot be negative
        predicted_val = max(0.0, predicted_val)
        sales_history.append(predicted_val)
        
        future_records.append({
            "Date": current_date,
            "MonthName": current_date.strftime("%B"),
            "Year": current_date.year,
            "ForecastedSales_INR": round(predicted_val, 2)
        })
        
    return pd.DataFrame(future_records)

# ==============================================================================
# 6. MAIN EXECUTION FLOW
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" SUPERMARKET SALES FORECASTING ENGINE - SUPERMARKETSALESDB ")
    print("=" * 70)
    
    try:
        conn = get_connection()
        df_raw = load_monthly_sales_data(conn)
        conn.close()
    except Exception:
        print("\n[FALLBACK MODE] Unable to connect to live SQL Server instance.")
        print("Tip: Run the provided SupermarketSalesDB.sql in SSMS first.")
        sys.exit(1)
        
    # Feature Engineering
    df_features = create_features(df_raw)
    
    # Train and compare
    results = train_and_evaluate_models(df_features, test_months=6)
    
    # Produce Future 6 Months Forecast using Random Forest
    rf_model = results["rf_model"]
    feature_cols = results["feature_cols"]
    df_forecast = forecast_future_sales(rf_model, df_features, feature_cols, horizon_months=6)
    
    print("\n=== PROJECTED FUTURE MONTHLY SALES (H1 2027) ===")
    print(df_forecast.to_string(index=False))
    print("\nForecast pipeline execution finished successfully!")
