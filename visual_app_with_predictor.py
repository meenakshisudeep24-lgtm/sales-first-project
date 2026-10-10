import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# 1. Page Configuration & Custom Theme Styling
st.set_page_config(page_title="Enterprise Predictive Visualizer", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size:28px; font-weight:bold; color: #1E3A8A; margin-bottom: 5px; }
    .predictor-box {
        background-color: #EFF6FF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #BFDBFE;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .metric-title { font-size: 13px; color: #64748B; font-weight: 600; }
    .metric-value { font-size: 22px; color: #1E293B; font-weight: bold; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# 2. Secure Data Ingestion Layer with Trim/Coalesce Fixes Built-in
@st.cache_data
def load_all_dashboard_data(db_path='SalesDB.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        CAST(i.InvoiceDate AS TEXT) as InvoiceDate,
        i.InvoiceID,
        cm.CustomerName,
        cm.City,
        TRIM(COALESCE(cm.State, 'Unknown')) as State,
        id.ProductID,
        p.ProductName,
        p.Category,
        id.Quantity,
        id.UnitPrice,
        id.LineTotal,
        i.TotalAmount,
        i.PaymentStatus
    FROM InvoiceDetails id
    JOIN Invoice i ON id.InvoiceID = i.InvoiceID
    JOIN ProductMaster p ON id.ProductID = p.ProductID
    JOIN CustomerMaster cm ON i.CustomerID = cm.CustomerID;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['State'] = df['State'].astype(str).str.strip().str.upper()
    return df

# Initialize base data streams
df = load_all_dashboard_data('SalesDB.db')
max_date = df['InvoiceDate'].max()

# Setup True Real-world Time parameters
real_world_today = pd.to_datetime(datetime.now().date())
real_world_tomorrow = real_world_today + pd.Timedelta(days=1)

# =====================================================================
# 🧠 BACKGROUND MACHINE LEARNING TRAIN PIPELINES
# =====================================================================
@st.cache_resource
def train_prediction_models(df):
    valid_df = df[df['PaymentStatus'] != 'Cancelled'].copy()
    valid_df['YearMonth'] = valid_df['InvoiceDate'].dt.to_period('M')
    
    # Model A: Macro Store Turnover Predictor setup
    store_monthly = valid_df.groupby('YearMonth')['TotalAmount'].sum().reset_index()
    store_monthly['Timestamp'] = store_monthly['YearMonth'].dt.to_timestamp()
    store_monthly['Year'] = store_monthly['Timestamp'].dt.year
    store_monthly['Month'] = store_monthly['Timestamp'].dt.month
    store_monthly['PrevMonthTurnover'] = store_monthly['TotalAmount'].shift(1).bfill()
    
    model_turnover = RandomForestRegressor(n_estimators=100, random_state=42)
    model_turnover.fit(store_monthly[['Year', 'Month', 'PrevMonthTurnover']], store_monthly['TotalAmount'])
    
    # Model B: Individual Product Demand Predictor setup
    prod_monthly = valid_df.groupby(['YearMonth', 'ProductID', 'ProductName']).agg({'Quantity': 'sum'}).reset_index()
    prod_monthly['Timestamp'] = prod_monthly['YearMonth'].dt.to_timestamp()
    prod_monthly['Year'] = prod_monthly['Timestamp'].dt.year
    prod_monthly['Month'] = prod_monthly['Timestamp'].dt.month
    prod_monthly = prod_monthly.sort_values(by=['ProductID', 'Timestamp'])
    prod_monthly['PrevMonthQty'] = prod_monthly.groupby('ProductID')['Quantity'].shift(1).bfill()
    prod_monthly['AvgHistQty'] = prod_monthly.groupby('ProductID')['Quantity'].transform('mean')
    
    model_product = RandomForestRegressor(n_estimators=100, random_state=42)
    model_product.fit(prod_monthly[['ProductID', 'Year', 'Month', 'PrevMonthQty', 'AvgHistQty']], prod_monthly['Quantity'])
    
    return model_turnover, model_product, store_monthly, prod_monthly

# FIX: Explicit alignment of matching variable names to prevent NameErrors
model_t, model_p, store_m_df, prod_m_df = train_prediction_models(df)

# =====================================================================
# 🖥️ STREAMLIT FRONTEND USER INTERFACE RENDER LAYER
# =====================================================================

st.markdown("<div class='main-header'>📈 Enterprise Sales Analytics & AI Predictive Suite</div>", unsafe_allow_html=True)
st.caption("Integrated Final Capstone Dashboard: Real-time Visualizations with live AI Scenario Forecasting Models")
st.markdown("---")

# 📊 TOP SECTION: MONTH FILTER PREDICTOR PANEL
st.markdown("<div class='predictor-box'>", unsafe_allow_html=True)
st.markdown("### 🔮 Predictive AI Simulation Filter Control")
st.write("Select a future timeline target window below to run live machine learning scenario projections:")

p_col1, p_col2 = st.columns(2)

with p_col1:
    future_months = [
        ("September 2026", 2026, 9),
        ("October 2026", 2026, 10),
        ("November 2026", 2026, 11),
        ("December 2026", 2026, 12)
    ]
    month_labels = [m[0] for m in future_months]
    selected_label = st.selectbox("🎯 Choose Target Forecast Month", month_labels, index=1) # Defaults to October 2026
    
    # Resolve numeric targets based on selections
    chosen_tuple = future_months[month_labels.index(selected_label)]
    t_year = chosen_tuple[1]
    t_month = chosen_tuple[2]

with p_col2:
    # RUN LIVE ML COMPUTATIONS
    last_actual_turnover = store_m_df['TotalAmount'].iloc[-1]
    input_macro = pd.DataFrame([{'Year': t_year, 'Month': t_month, 'PrevMonthTurnover': last_actual_turnover}])
    pred_store_rev = model_t.predict(input_macro)[0]
    
    # Compute product demand trends for that chosen month target matrix
    unique_products = prod_m_df[['ProductID', 'ProductName', 'AvgHistQty']].drop_duplicates(subset=['ProductID']).copy()
    last_m_actuals = prod_m_df[prod_m_df['Timestamp'] == prod_m_df['Timestamp'].max()][['ProductID', 'Quantity']]
    last_m_actuals = last_m_actuals.rename(columns={'Quantity': 'PrevMonthQty'})
    next_p_df = unique_products.merge(last_m_actuals, on='ProductID', how='left').fillna(0)
    next_p_df['Year'] = t_year
    next_p_df['Month'] = t_month
    
    next_p_df['PredQty'] = model_p.predict(next_p_df[['ProductID', 'Year', 'Month', 'PrevMonthQty', 'AvgHistQty']])
    top_winning_product = next_p_df.sort_values(by='PredQty', ascending=False).iloc[0]
    
    # Output the AI results cleanly inside the colored callout row
    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.metric(label=f"Projected Store Revenue ({selected_label})", value=f"₹{pred_store_rev:,.2f}")
    with r_col2:
        st.metric(label=f"Predicted Top Product Demand", value=f"{top_winning_product['ProductName']} ({round(top_winning_product['PredQty'], 1)} units)")

st.markdown("</div>", unsafe_allow_html=True)

# 🎛️ SIDEBAR CONTROLS: FOR HISTORICAL DESK ANALYSIS ONLY
st.sidebar.header("🗂️ Historical Dashboard Filters")
status_options = df['PaymentStatus'].unique().tolist()
selected_status = st.sidebar.multiselect("Filter Historical Charts by Payment Status", status_options, default=['Paid', 'Pending', 'Overdue'])

category_options = df['Category'].unique().tolist()
selected_categories = st.sidebar.multiselect("Filter Historical Charts by Category", category_options, default=category_options)

# Filtering baseline operations dataframe layout masks
filtered_df = df[(df['PaymentStatus'].isin(selected_status)) & (df['Category'].isin(selected_categories))]

# MIDDLE SECTION: HISTORICAL METRIC TILES
st.markdown("### 📊 Current Historical Performance Overview")
if not filtered_df.empty:
    total_rev = filtered_df['LineTotal'].sum()
    total_units = filtered_df['Quantity'].sum()
    total_orders = filtered_df['InvoiceID'].nunique()
    avg_order_val = total_rev / total_orders if total_orders > 0 else 0

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>💰 HISTORICAL REVENUE</div><div class='metric-value'>₹{total_rev:,.2f}</div></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>📦 HISTORICAL UNITS SOLD</div><div class='metric-value'>{total_units:,} Items</div></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>🧾 TOTAL HISTORICAL INVOICES</div><div class='metric-value'>{total_orders:,} Orders</div></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>⚖️ HISTORICAL AVERAGE ORDER VALUE</div><div class='metric-value'>₹{avg_order_val:,.2f}</div></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No historical data available matching the criteria selected in the sidebar filters.")
    st.stop()

# BOTTOM SECTION: CHARTS AND TABLES DISPLAY LAYER
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Historical Sales Trends Over Time")
    timeline_df = filtered_df.groupby('InvoiceDate')['LineTotal'].sum().reset_index().set_index('InvoiceDate')
    st.line_chart(timeline_df['LineTotal'], color="#3B82F6")

with chart_col2:
    st.subheader("📊 Category Wise Revenue Shares")
    cat_df = filtered_df.groupby('Category')['LineTotal'].sum().reset_index().set_index('Category')
    st.bar_chart(cat_df['LineTotal'], color="#10B981")

st.markdown("---")
view_col1, view_col2 = st.columns(2)

with view_col1:
    st.subheader("🏆 Product Standings Ranking (Top Volume)")
    top_prods = filtered_df.groupby('ProductName').agg({
        'Quantity': 'sum',
        'LineTotal': 'sum'
    }).reset_index().sort_values(by='Quantity', ascending=False).head(10)
    st.dataframe(top_prods.rename(columns={'Quantity': 'Total Units Sold', 'LineTotal': 'Total Revenue'}), use_container_width=True)        