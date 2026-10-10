"""
========================================================================================
Streamlit Application: Supermarket Sales Intelligence & ML Forecasting
Command to run: streamlit run streamlit_app.py
========================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pyodbc
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

# 1. PAGE SETUP
st.set_page_config(
    page_title="Supermarket Sales Forecasting AI Studio",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stMetric {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# 2. SIDEBAR CONFIGURATION
st.sidebar.image("https://images.unsplash.com/photo-1542838132-92c53300491e?w=600&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🛒 SupermarketSalesDB")
st.sidebar.caption("Machine Learning Sales Forecasting Suite")

data_source = st.sidebar.radio("Data Connection:", ["Live SQL Server (pyodbc)", "Synthetic Sample Data (Offline Mode)"])

server_name = r"localhostSQLEXPRESS"
db_name = "SupermarketSalesDB"
driver = "{ODBC Driver 17 for SQL Server}"
trusted = True

if data_source == "Live SQL Server (pyodbc)":
    st.sidebar.markdown("### 🔌 SQL Server Settings")
    server_preset = st.sidebar.selectbox(
        "Server Instance Preset:",
        [
            r"localhostSQLEXPRESS (SQL Server Express Default)",
            r".SQLEXPRESS (Alternative Local Express)",
            "localhost (Developer / Standard Edition)",
            "127.0.0.1,1433 (Direct TCP/IP Port 1433)",
            "Custom Instance Name"
        ]
    )
    if "Custom" in server_preset:
        server_name = st.sidebar.text_input("Enter SQL Server Instance:", value=r"localhostSQLEXPRESS")
    elif "localhost\SQLEXPRESS" in server_preset:
        server_name = r"localhostSQLEXPRESS"
    elif ".\SQLEXPRESS" in server_preset:
        server_name = r".SQLEXPRESS"
    elif "127.0.0.1" in server_preset:
        server_name = "127.0.0.1,1433"
    else:
        server_name = "localhost"

    db_name = st.sidebar.text_input("Database Name:", value="SupermarketSalesDB")
    driver = st.sidebar.selectbox("ODBC Driver:", ["{ODBC Driver 17 for SQL Server}", "{ODBC Driver 18 for SQL Server}", "{SQL Server}"])
    trusted = st.sidebar.checkbox("Windows Authentication", value=True)

st.sidebar.subheader("⚙️ Model Hyperparameters")
selected_model = st.sidebar.selectbox("Forecasting Algorithm:", ["Random Forest Regressor", "Linear Regression", "Both (Ensemble / Comparison)"])
test_size_months = st.sidebar.slider("Test Hold-Out Size (Months):", min_value=3, max_value=8, value=6)
forecast_horizon = st.sidebar.slider("Forecast Horizon Ahead (Months):", min_value=3, max_value=12, value=6)
include_festive = st.sidebar.checkbox("Include Indian Festive Season Flag (Diwali)", value=True)

# 3. DATA INGESTION FUNCTION
@st.cache_data(ttl=600)
def load_data(use_live_sql, srv=r"localhostSQLEXPRESS", db="SupermarketSalesDB", drv="{ODBC Driver 17 for SQL Server}", win_auth=True):
    if use_live_sql:
        # Try candidate connection strings to resolve Error 08001 automatically
        candidates = [srv, r"localhostSQLEXPRESS", r".SQLEXPRESS", "localhost", "127.0.0.1,1433", r"(localdb)mssqllocaldb"]
        seen_cand = set()
        candidate_list = [c for c in candidates if not (c in seen_cand or seen_cand.add(c))]

        drivers_list = [drv, "{ODBC Driver 17 for SQL Server}", "{ODBC Driver 18 for SQL Server}", "{SQL Server}"]
        seen_drv = set()
        driver_list = [d for d in drivers_list if not (d in seen_drv or seen_drv.add(d))]

        last_err = None
        for cand_srv in candidate_list:
            for cand_drv in driver_list:
                try:
                    conn_str = f"DRIVER={cand_drv};SERVER={cand_srv};DATABASE={db};Trusted_Connection={'yes' if win_auth else 'no'};TrustServerCertificate=yes;"
                    conn = pyodbc.connect(conn_str, timeout=3)
                    df = pd.read_sql("SELECT [Year], [Month], [MonthName], CAST(TotalSales AS FLOAT) AS TotalSales, TotalQuantity, NumberOfTransactions FROM dbo.MonthlySales ORDER BY [Year], [Month];", conn)
                    conn.close()
                    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
                    return df, f"Connected live to SQL Server instance '{cand_srv}' via {cand_drv}!"
                except Exception as e:
                    last_err = e

        st.sidebar.error(f"Live SQL Server connection failed: {last_err}")
        with st.sidebar.expander("🛠️ How to fix Error 08001 (Named Pipes / Instance Error)"):
            st.markdown("""
            **1. Check Instance Name in SSMS:**
            When you open SSMS, look at the **Server name** field. If it says 'DESKTOP-XYZ\\SQLEXPRESS' or '.\\SQLEXPRESS', select **Custom Instance Name** and enter that exact name.

            **2. Enable TCP/IP in SQL Server Configuration Manager:**
            - Open **SQL Server Configuration Manager**.
            - Expand **SQL Server Network Configuration** -> **Protocols for SQLEXPRESS**.
            - Double-click **TCP/IP** -> set **Enabled** to **Yes**.
            - Restart the **SQL Server (SQLEXPRESS)** service in Windows Services.

            **3. Start SQL Server Browser Service:**
            - In Windows Services (services.msc), ensure **SQL Server Browser** is **Running** (startup type: Automatic).
            """)
        st.sidebar.info("Falling back to realistic pre-seeded dataset for demo.")

    # High-fidelity realistic 24 months fallback matching SupermarketSalesDB
    base_sales = [
        # 2025
        (2025, 1, "January", 710000, 4820, 940),
        (2025, 2, "February", 645000, 4380, 860),
        (2025, 3, "March", 735000, 5010, 970),
        (2025, 4, "April", 685000, 4650, 910),
        (2025, 5, "May", 720000, 4890, 950),
        (2025, 6, "June", 695000, 4720, 920),
        (2025, 7, "July", 740000, 5050, 980),
        (2025, 8, "August", 760000, 5160, 1010),
        (2025, 9, "September", 730000, 4960, 970),
        (2025, 10, "October", 945000, 6420, 1260), # Diwali surge
        (2025, 11, "November", 985000, 6680, 1310), # Diwali surge
        (2025, 12, "December", 840000, 5710, 1120),
        # 2026 (YoY Organic Growth ~14%)
        (2026, 1, "January", 815000, 5540, 1080),
        (2026, 2, "February", 742000, 5040, 990),
        (2026, 3, "March", 845000, 5750, 1130),
        (2026, 4, "April", 790000, 5370, 1050),
        (2026, 5, "May", 830000, 5640, 1110),
        (2026, 6, "June", 805000, 5480, 1070),
        (2026, 7, "July", 855000, 5820, 1140),
        (2026, 8, "August", 880000, 5980, 1170),
        (2026, 9, "September", 845000, 5740, 1130),
        (2026, 10, "October", 1090000, 7410, 1450), # Diwali surge
        (2026, 11, "November", 1140000, 7750, 1520), # Diwali surge
        (2026, 12, "December", 975000, 6630, 1300)
    ]
    df = pd.DataFrame(base_sales, columns=["Year", "Month", "MonthName", "TotalSales", "TotalQuantity", "NumberOfTransactions"])
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-01")
    return df, "Loaded seeded SupermarketSalesDB dataset (24 Months)"

is_live = (data_source == "Live SQL Server (pyodbc)")
df_sales, conn_status = load_data(is_live, srv=server_name, db=db_name, drv=driver, win_auth=trusted)

# 4. APP HEADER & KPIS
st.title("📊 Supermarket Sales Forecasting Studio")
st.caption(f"Status: {conn_status} | Target: 24 Months (Jan 2025 – Dec 2026)")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_rev = df_sales["TotalSales"].sum()
total_units = df_sales["TotalQuantity"].sum()
total_tx = df_sales["NumberOfTransactions"].sum()
avg_monthly = df_sales["TotalSales"].mean()

kpi1.metric("Total Revenue (24M)", f"₹{total_rev:,.0f}")
kpi2.metric("Total Units Sold", f"{total_units:,.0f}")
kpi3.metric("Total Transactions", f"{total_tx:,.0f}")
kpi4.metric("Average Monthly Sales", f"₹{avg_monthly:,.0f}")

st.markdown("---")

# 5. FEATURE ENGINEERING
def build_features(data):
    df_f = data.copy()
    df_f["TimeIndex"] = np.arange(1, len(df_f) + 1)
    df_f["Quarter"] = df_f["Date"].dt.quarter
    df_f["Month_Sin"] = np.sin(2 * np.pi * df_f["Month"] / 12.0)
    df_f["Month_Cos"] = np.cos(2 * np.pi * df_f["Month"] / 12.0)
    df_f["IsFestiveSeason"] = df_f["Month"].isin([10, 11]).astype(int) if include_festive else 0
    df_f["Sales_Lag_1"] = df_f["TotalSales"].shift(1)
    df_f["Sales_Lag_2"] = df_f["TotalSales"].shift(2)
    df_f["Rolling_Mean_3"] = df_f["TotalSales"].shift(1).rolling(window=3).mean()
    return df_f.dropna().reset_index(drop=True)

df_feat = build_features(df_sales)
feature_cols = ["TimeIndex", "Month_Sin", "Month_Cos", "Quarter", "IsFestiveSeason", "Sales_Lag_1", "Sales_Lag_2", "Rolling_Mean_3"]

split_idx = len(df_feat) - test_size_months
train_data = df_feat.iloc[:split_idx]
test_data = df_feat.iloc[split_idx:]

X_train, y_train = train_data[feature_cols], train_data["TotalSales"]
X_test, y_test = test_data[feature_cols], test_data["TotalSales"]

# Train Models
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_test_preds = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_test_preds = rf.predict(X_test)

# Metrics
def get_metrics_dict(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE (INR ₹)": f"₹{rmse:,.0f}", "MAE (INR ₹)": f"₹{mae:,.0f}", "MAPE (%)": f"{mape:.2f}%", "R² Score": f"{r2:.4f}"}

tab1, tab2, tab3 = st.tabs(["📈 Forecast Engine", "🔍 Model Evaluation & Metrics", "📋 Raw Historical Data"])

with tab1:
    st.subheader("Historical Sales & Future Multi-Step Projection")
    
    # Generate multi-step future projection
    future_hist = list(df_feat["TotalSales"].values)
    future_dates = []
    future_preds_rf = []
    future_preds_lr = []
    
    last_d = df_feat["Date"].iloc[-1]
    curr_t = df_feat["TimeIndex"].iloc[-1]
    
    for _ in range(forecast_horizon):
        if last_d.month == 12:
            last_d = pd.to_datetime(f"{last_d.year + 1}-01-01")
        else:
            last_d = pd.to_datetime(f"{last_d.year}-{last_d.month + 1:02d}-01")
        curr_t += 1
        future_dates.append(last_d)
        
        m = last_d.month
        q = (m - 1) // 3 + 1
        m_sin = np.sin(2 * np.pi * m / 12.0)
        m_cos = np.cos(2 * np.pi * m / 12.0)
        fest = 1 if (m in [10, 11] and include_festive) else 0
        l1 = future_hist[-1]
        l2 = future_hist[-2]
        r3 = np.mean([l1, l2, future_hist[-3]])
        
        f_vec = pd.DataFrame([{
            "TimeIndex": curr_t, "Month_Sin": m_sin, "Month_Cos": m_cos, "Quarter": q,
            "IsFestiveSeason": fest, "Sales_Lag_1": l1, "Sales_Lag_2": l2, "Rolling_Mean_3": r3
        }])[feature_cols]
        
        rf_p = max(0.0, rf.predict(f_vec)[0])
        lr_p = max(0.0, lr.predict(f_vec)[0])
        future_hist.append(rf_p)
        future_preds_rf.append(rf_p)
        future_preds_lr.append(lr_p)
        
    fig = go.Figure()
    
    # Historical Actual
    fig.add_trace(go.Scatter(
        x=df_sales["Date"], y=df_sales["TotalSales"],
        mode="lines+markers", name="Actual Sales (Historical)",
        line=dict(color="#38BDF8", width=3),
        marker=dict(size=6)
    ))
    
    # Test Predictions
    fig.add_trace(go.Scatter(
        x=test_data["Date"], y=rf_test_preds,
        mode="lines+markers", name="Random Forest (Test Prediction)",
        line=dict(color="#A855F7", width=2, dash="dot"),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_data["Date"], y=lr_test_preds,
        mode="lines+markers", name="Linear Regression (Test Prediction)",
        line=dict(color="#F59E0B", width=2, dash="dash"),
        marker=dict(size=6)
    ))
    
    # Future Forecast (Random Forest)
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_preds_rf,
        mode="lines+markers", name="Future Forecast (Random Forest)",
        line=dict(color="#10B981", width=3, dash="dashdot"),
        marker=dict(size=8, symbol="star")
    ))
    
    fig.update_layout(
        title="Supermarket Monthly Revenue: Actual vs Test Validation vs Future Forecast",
        xaxis_title="Timeline",
        yaxis_title="Total Sales (INR ₹)",
        template="plotly_dark",
        hovermode="x unified",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Future table
    df_future_view = pd.DataFrame({
        "Month": [d.strftime("%B %Y") for d in future_dates],
        "Random Forest Forecast (INR)": [f"₹{p:,.2f}" for p in future_preds_rf],
        "Linear Regression Forecast (INR)": [f"₹{p:,.2f}" for p in future_preds_lr],
    })
    st.subheader(f"🔮 Future {forecast_horizon}-Month Forecast Table")
    st.table(df_future_view)

with tab2:
    st.subheader("Model Performance Comparison (Test Hold-Out Set)")
    m_df = pd.DataFrame([
        get_metrics_dict("Random Forest Regressor", y_test, rf_test_preds),
        get_metrics_dict("Linear Regression", y_test, lr_test_preds)
    ])
    st.dataframe(m_df, use_container_width=True)
    
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.subheader("🌲 Random Forest Feature Importance")
        fi = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig_fi = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Predictor Importance", template="plotly_dark")
        st.plotly_chart(fig_fi, use_container_width=True)
        
    with col_feat2:
        st.subheader("📐 Linear Regression Coefficients")
        coef_df = pd.DataFrame({
            "Feature": feature_cols,
            "Coefficient": lr.coef_
        }).sort_values("Coefficient", ascending=False)
        fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="Regression Weights", template="plotly_dark")
        st.plotly_chart(fig_coef, use_container_width=True)

with tab3:
    st.subheader("Monthly Sales Historical View (from dbo.MonthlySales)")
    st.dataframe(df_sales, use_container_width=True)
