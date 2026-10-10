import pandas as pd
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

db_name = 'SalesDB.db'

# 1. Connect to our local SQLite database
print("Connecting to SalesDB.db to extract features...")
conn = sqlite3.connect(db_name)

# 2. Query sales data grouped by date to predict daily revenue
query = """
    SELECT InvoiceDate, SUM(TotalAmount) as DailySales
    FROM Invoice
    GROUP BY InvoiceDate
    ORDER BY InvoiceDate ASC;
"""
df = pd.read_sql_query(query, conn)
conn.close()

print(f" Extracted {len(df)} days of historical sales data.")

# 3. Feature Engineering: Convert dates into numeric sequence days (0, 1, 2...)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['DayIndex'] = (df['InvoiceDate'] - df['InvoiceDate'].min()).dt.days

# Define features (X) and target label (y)
X = df[['DayIndex']] # Independent variable (Time sequence)
y = df['DailySales']  # Dependent variable (Revenue to predict)

# 4. Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and train the Linear Regression Model
print("Training the Machine Learning Prediction model...")
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate model accuracy performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Performance Evaluation ---")
print(f"R2 Score (Accuracy): {r2:.4f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")

# 7. Make a Proactive Future Sales Prediction
last_day = df['DayIndex'].max()
next_week_days = np.array([[last_day + 1], [last_day + 7]])
predictions = model.predict(next_week_days)

print("\n--- Sales Revenue Predictions ---")
print(f"Predicted revenue for tomorrow: ${predictions[0]:.2f}")
print(f"Predicted revenue for next week: ${predictions[1]:.2f}")

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

def run_prediction_suite(db_path='SalesDB.db'):
    # ==========================================
    # DATA EXTRACTION LAYER
    # ==========================================
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        return f"❌ Connection Error: Ensure {db_path} is in this exact folder. Details: {e}"
        
    # Extract base transactional data
    tx_query = """
    SELECT 
        CAST(i.InvoiceDate AS TEXT) as InvoiceDate,
        i.InvoiceID,
        i.CustomerID,
        cm.CustomerName,
        i.SalesPersonID,
        id.ProductID,
        p.ProductName,
        id.Quantity,
        id.LineTotal
    FROM InvoiceDetails id
    JOIN Invoice i ON id.InvoiceID = i.InvoiceID
    JOIN ProductMaster p ON id.ProductID = p.ProductID
    JOIN CustomerMaster cm ON i.CustomerID = cm.CustomerID
    WHERE i.PaymentStatus != 'Cancelled';
    """
    df = pd.read_sql_query(tx_query, conn)
    conn.close()

    if df.empty:
        return "❌ The transaction table is empty or could not be retrieved."

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    max_date = df['InvoiceDate'].max()

    print(f"📊 Analyzing {len(df)} transactions spanning up to {max_date.strftime('%Y-%m-%d')}...\n")

    # ==========================================
    # PREDICTION 1: TOP SELLING PRODUCT FORECAST
    # ==========================================
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    prod_monthly = df.groupby(['YearMonth', 'ProductID', 'ProductName']).agg({'Quantity': 'sum'}).reset_index()
    prod_monthly['Timestamp'] = prod_monthly['YearMonth'].dt.to_timestamp()
    
    prod_monthly['Year'] = prod_monthly['Timestamp'].dt.year
    prod_monthly['Month'] = prod_monthly['Timestamp'].dt.month
    prod_monthly = prod_monthly.sort_values(by=['ProductID', 'Timestamp'])
    prod_monthly['PrevMonthQty'] = prod_monthly.groupby('ProductID')['Quantity'].shift(1).fillna(0)
    prod_monthly['AvgHistQty'] = prod_monthly.groupby('ProductID')['Quantity'].transform('mean')

    X_p = prod_monthly[['ProductID', 'Year', 'Month', 'PrevMonthQty', 'AvgHistQty']]
    y_p = prod_monthly['Quantity']
    
    model_p = RandomForestRegressor(n_estimators=100, random_state=42)
    model_p.fit(X_p, y_p)

    next_m = max_date + pd.offsets.MonthBegin(1)
    next_p_df = prod_monthly[['ProductID', 'ProductName', 'AvgHistQty']].drop_duplicates(subset=['ProductID']).copy()
    last_m_actuals = prod_monthly[prod_monthly['Timestamp'] == prod_monthly['Timestamp'].max()][['ProductID', 'Quantity']]
    last_m_actuals = last_m_actuals.rename(columns={'Quantity': 'PrevMonthQty'})
    
    next_p_df = next_p_df.merge(last_m_actuals, on='ProductID', how='left').fillna(0)
    next_p_df['Year'] = next_m.year
    next_p_df['Month'] = next_m.month
    
    next_p_df['PredQty'] = model_p.predict(next_p_df[['ProductID', 'Year', 'Month', 'PrevMonthQty', 'AvgHistQty']])
    top_prod = next_p_df.sort_values(by='PredQty', ascending=False).iloc[0]

    # ==========================================
    # PREDICTION 2: TOTAL STORE REVENUE FORECASTING
    # ==========================================
    daily_rev = df.groupby(df['InvoiceDate'].dt.date)['LineTotal'].sum().reset_index()
    daily_rev['InvoiceDate'] = pd.to_datetime(daily_rev['InvoiceDate'])
    
    daily_rev['DayOfWeek'] = daily_rev['InvoiceDate'].dt.dayofweek
    daily_rev['DayOfMonth'] = daily_rev['InvoiceDate'].dt.day
    daily_rev['Rolling7DayAvg'] = daily_rev['LineTotal'].rolling(7, min_periods=1).mean()

    X_r = daily_rev[['DayOfWeek', 'DayOfMonth', 'Rolling7DayAvg']]
    y_r = daily_rev['LineTotal']

    model_r = RandomForestRegressor(n_estimators=100, random_state=42)
    model_r.fit(X_r, y_r)

    last_rolling_avg = daily_rev['LineTotal'].tail(7).mean()
    tomorrow_features = pd.DataFrame([{
        'DayOfWeek': (max_date.dayofweek + 1) % 7,
        'DayOfMonth': (max_date + pd.Timedelta(days=1)).day,
        'Rolling7DayAvg': last_rolling_avg
    }])
    
    pred_tomorrow_rev = model_r.predict(tomorrow_features)[0]

    # ==========================================
    # PREDICTION 3: CUSTOMER CHURN RISK ASSIGNMENT
    # ==========================================
    cust_profile = df.groupby(['CustomerID', 'CustomerName']).agg(
        LastPurchase=('InvoiceDate', 'max'),
        TotalOrders=('InvoiceID', 'nunique'),
        TotalSpent=('LineTotal', 'sum')
    ).reset_index()

    cust_profile['DaysSinceLastPurchase'] = (max_date - cust_profile['LastPurchase']).dt.days
    
    # Establish a structural heuristic baseline target: 
    # If a customer hasn't ordered in over 60 days, flag them as historical churn
    cust_profile['Churned'] = np.where(cust_profile['DaysSinceLastPurchase'] > 60, 1, 0)

    X_c = cust_profile[['TotalOrders', 'TotalSpent']]
    y_c = cust_profile['Churned']

    # Use a Classifier to score probabilities of accounts becoming inactive
    model_c = RandomForestClassifier(n_estimators=100, random_state=42)
    model_c.fit(X_c, y_c)

    cust_profile['ChurnProbability'] = model_c.predict_proba(X_c)[:, 1]
    at_risk_customers = cust_profile[cust_profile['Churned'] == 0].sort_values(by='ChurnProbability', ascending=False)

    # ==========================================
    # EXECUTIVE DASHBOARD GENERATION TERMINAL VIEW
    # ==========================================
    print("=====================================================================")
    print(" 🛠️  ENTERPRISE SALES PREDICTIVE ANALYTICS EXECUTIVE REPORT")
    print("=====================================================================\n")
    
    print("📋 [ pillar 1: PRODUCT DEMAND FORECAST ]")
    print(f" ➡️  Top Selling Product Prediction for {next_m.strftime('%B %Y')}:")
    print(f"    🌟 Product: '{top_prod['ProductName']}' (ID: {int(top_prod['ProductID'])})")
    print(f"    📦 Projected Demand: {round(top_prod['PredQty'], 1)} units\n")
    print("---------------------------------------------------------------------")
    
    print("💰 [ pillar 2: FINANCIAL REVENUE PROJECTIONS ]")
    print(f" ➡️  Predicted Revenue for tomorrow ({ (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d') }):")
    print(f"    💵 Total Projected Sales: ${round(pred_tomorrow_rev, 2)}\n")
    print("---------------------------------------------------------------------")
    
    print("⚠️  [ pillar 3: CUSTOMER RETENTION RISK MATRIX ]")
    print(" ➡️  Top 3 Active High-Value Accounts showing high risk metrics:")
    for idx, row in at_risk_customers.head(3).iterrows():
        print(f"    🔴 Customer: {row['CustomerName']} | At Risk Probability: {round(row['ChurnProbability'] * 100, 1)}% (Spent: ${round(row['TotalSpent'], 2)})")
    print("=====================================================================")

if __name__ == "__main__":
    run_prediction_suite('SalesDB.db')


import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predict_october_turnover(db_path='SalesDB.db'):
    # 1. Connect directly to your local database engine
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        return f"❌ Database Connection Error: {e}"
    
    # 2. Extract transactional totals grouped by invoice date
    query = """
    SELECT 
        CAST(InvoiceDate AS TEXT) as InvoiceDate,
        SUM(TotalAmount) AS DailyRevenue
    FROM Invoice
    WHERE PaymentStatus != 'Cancelled'
    GROUP BY InvoiceDate;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return "❌ No transactional historical data found in the database."
        
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 3. Resample the daily data into macro-level Month periods
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly_data = df.groupby('YearMonth')['DailyRevenue'].sum().reset_index()
    monthly_data = monthly_data.rename(columns={'DailyRevenue': 'MonthlyTurnover'})
    
    # Sort chronologically to preserve time sequence properties
    monthly_data = monthly_data.sort_values(by='YearMonth')
    monthly_data['Timestamp'] = monthly_data['YearMonth'].dt.to_timestamp()
    
    # 4. Feature Engineering: Extract structural dates & rolling momentum lags
    monthly_data['Year'] = monthly_data['Timestamp'].dt.year
    monthly_data['Month'] = monthly_data['Timestamp'].dt.month
    
    # Create lag features (previous month's total) to capture the operational path trajectory
    monthly_data['PrevMonthTurnover'] = monthly_data['MonthlyTurnover'].shift(1).bfill()

    # 5. Define ML feature frameworks
    features = ['Year', 'Month', 'PrevMonthTurnover']
    X = monthly_data[features]
    y = monthly_data['MonthlyTurnover']
    
    # 6. Train the machine learning forecaster
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 7. Construct target feature row for October 2026 explicitly
    # We grab the last actual monthly record to pass down as the predictive momentum vector
    last_actual_turnover = monthly_data['MonthlyTurnover'].iloc[-1]
    
    october_features = pd.DataFrame([{
        'Year': 2026,
        'Month': 10,
        'PrevMonthTurnover': last_actual_turnover
    }])
    
    # 8. Compute the financial forecast
    predicted_turnover = model.predict(october_features)[0]
    
    # Print out detailed telemetry data metrics view
    print("=========================================================")
    print(" 📊 EXECUTIVE STRATEGIC TARGET MONTHLY TURNOVER FORECAST")
    print("=========================================================\n")
    print(f" 🗓️ Target Window : October 2026")
    print(f" 📈 Latest Monthly Anchor Run: ${last_actual_turnover:,.2f}\n")
    print(f" 💵 PROJECTED OCTOBER 2026 TURNOVER: ${predicted_turnover:,.2f}")
    print("=========================================================")
    
    return predicted_turnover

if __name__ == "__main__":
    predict_october_turnover('SalesDB.db')
