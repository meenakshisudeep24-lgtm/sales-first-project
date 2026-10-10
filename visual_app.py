import streamlit as st
import sqlite3
import pandas as pd
import numpy as np

# 1. Page Configuration & Custom Theme Styling
st.set_page_config(page_title="Executive Sales Visualizer", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-title { font-size: 14px; color: #64748B; font-weight: 600; }
    .metric-value { font-size: 24px; color: #1E293B; font-weight: bold; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# 2. Secure Data Ingestion Layer
def load_visual_data(db_path='SalesDB.db'):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        CAST(i.InvoiceDate AS TEXT) as InvoiceDate,
        i.InvoiceID,
        cm.CustomerName,
        cm.City,
        cm.State,
        p.ProductName,
        p.Category,
        id.Quantity,
        id.UnitPrice,
        id.LineTotal,
        i.PaymentStatus
    FROM InvoiceDetails id
    JOIN Invoice i ON id.InvoiceID = i.InvoiceID
    JOIN ProductMaster p ON id.ProductID = p.ProductID
    JOIN CustomerMaster cm ON i.CustomerID = cm.CustomerID;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

# Initialize Data
df = load_visual_data('SalesDB.db')

# 3. Header Panel
st.title("📈 Executive Sales Performance & Visual Dashboard")
st.caption("Real-Time Data Streams extracted directly from Enterprise Relational Databases")
st.markdown("---")

# 4. Interactive Interactive Sidebar Filters
st.sidebar.header("🎛️ Dashboard Control Filters")

# Filter 1: Order Status Component
status_options = df['PaymentStatus'].unique().tolist()
selected_status = st.sidebar.multiselect("Filter by Payment Status", status_options, default=['Paid', 'Pending', 'Overdue'])

# Filter 2: Product Category Component
category_options = df['Category'].unique().tolist()
selected_categories = st.sidebar.multiselect("Filter by Product Category", category_options, default=category_options)

# Apply Filter Masks to dataset
filtered_df = df[
    (df['PaymentStatus'].isin(selected_status)) & 
    (df['Category'].isin(selected_categories))
]

# 5. Core KPI Metric Summary Row
if not filtered_df.empty:
    total_rev = filtered_df['LineTotal'].sum()
    total_units = filtered_df['Quantity'].sum()
    total_orders = filtered_df['InvoiceID'].nunique()
    avg_order_val = total_rev / total_orders if total_orders > 0 else 0

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>💰 TOTAL GROSS REVENUE</div><div class='metric-value'>₹{total_rev:,.2f}</div></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>📦 TOTAL UNITS SOLD</div><div class='metric-value'>{total_units:,} Items</div></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>🧾 TOTAL INVOICES RUN</div><div class='metric-value'>{total_orders:,} Orders</div></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>📊 AVERAGE ORDER VALUE</div><div class='metric-value'>₹{avg_order_val:,.2f}</div></div>", unsafe_allow_html=True)
else:
    st.warning("⚠️ No records match current sidebar filter constraints.")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# 6. Data Visualization Layout Layer
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("📈 Chronological Sales Revenue Trend")
    # Aggregate revenue data down into clean continuous daily blocks
    timeline_df = filtered_df.groupby('InvoiceDate')['LineTotal'].sum().reset_index()
    timeline_df = timeline_df.set_index('InvoiceDate')
    
    # Renders an interactive native time-series line chart
    st.line_chart(timeline_df['LineTotal'], color="#3B82F6")

with chart_col2:
    st.subheader("🍕 Market Share Breakdown by Product Category")
    # Group item quantities by category
    cat_df = filtered_df.groupby('Category')['LineTotal'].sum().reset_index()
    cat_df = cat_df.set_index('Category')
    
    # Renders a comparative structural visual breakdown chart
    st.bar_chart(cat_df['LineTotal'], color="#10B981")

st.markdown("---")
view_col1, view_col2 = st.columns([3, 2])

with view_col1:
    st.subheader("🏆 Top Performing Products Matrix")
    # Isolate product volume counts
    top_prods = filtered_df.groupby('ProductName').agg({
        'Quantity': 'sum',
        'LineTotal': 'sum'
    }).rename(columns={'Quantity': 'Units Sold', 'LineTotal': 'Total Sales (₹)'}).sort_values(by='Units Sold', ascending=False).head(8)
    
    st.dataframe(top_prods, use_container_width=True)

with view_col2:
    st.subheader("📍 Regional Performance Rankings (Top States)")
    state_df = filtered_df.groupby('State')['LineTotal'].sum().reset_index()
    state_df = state_df.sort_values(by='LineTotal', ascending=False).rename(columns={'LineTotal': 'Revenue Generated (₹)'})
    
    st.dataframe(state_df, use_container_width=True, hide_index=True)
