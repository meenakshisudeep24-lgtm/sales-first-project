import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pyodbc

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Sales Prediction & Future Forecast Dashboard")
st.caption(
    "Historical sales analysis + multiple linear regression "
    "(trend + seasonality) monthly forecasting using your SalesDB."
)

# ---------------------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_sales(server, database, username, password, driver):

    if username:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={username};"
            f"PWD={password};"
            f"TrustServerCertificate=yes;"
        )
    else:
        conn_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"Trusted_Connection=yes;"
            f"TrustServerCertificate=yes;"
        )

    conn = pyodbc.connect(conn_str)

    query = """
    SELECT
        i.InvoiceID,
        CAST(i.InvoiceDate AS date) AS InvoiceDate,
        i.CustomerID,
        i.SalesPersonID,
        i.TotalAmount,
        i.PaymentStatus,
        d.ProductID,
        p.ProductName,
        p.Category,
        d.Quantity,
        d.UnitPrice,
        d.DiscountPct,
        d.LineTotal
    FROM dbo.Invoice i
    INNER JOIN dbo.InvoiceDetails d
        ON i.InvoiceID = d.InvoiceID
    LEFT JOIN dbo.ProductMaster p
        ON d.ProductID = p.ProductID
    ORDER BY i.InvoiceDate;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    for col in ["TotalAmount", "Quantity", "LineTotal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Category"] = df["Category"].fillna("Unknown")
    df["ProductName"] = df["ProductName"].fillna("Unknown")

    return df


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Database Settings")

    server = st.text_input("SQL Server", r"localhost\SQLEXPRESS")
    database = st.text_input("Database", "SalesDB")
    driver = st.text_input("ODBC Driver", "ODBC Driver 17 for SQL Server")

    auth = st.radio("Authentication", ["Windows", "SQL Server"], index=0)

    username = ""
    password = ""

    if auth == "SQL Server":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

    st.divider()

    st.header("🔮 Forecast Settings")

    forecast_months = st.slider(
        "Future months to predict",
        min_value=1,
        max_value=24,
        value=6,
    )

    metric = st.selectbox("Prediction metric", ["Sales", "Quantity"])

    exclude_last = st.checkbox(
        "Exclude latest month (may be incomplete)",
        value=True,
        help="A partly finished month looks artificially low and "
             "distorts the trend and the accuracy score.",
    )

    st.divider()

    st.caption(
        r"Tip: if your SQL Server is local, try "
        r"`localhost\SQLEXPRESS` or `.`."
    )


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

try:
    df = load_sales(server, database, username, password, driver)

except Exception as e:
    st.error("Could not connect to SQL Server.")
    st.code(str(e))
    st.info(
        "Check the SQL Server name, database name, ODBC driver, "
        "and authentication. Your database should be named SalesDB."
    )
    st.stop()


if df.empty:
    st.warning("No sales records were found.")
    st.stop()


# ---------------------------------------------------------------------
# PREPARE MONTHLY DATA
# ---------------------------------------------------------------------

monthly = (
    df.set_index("InvoiceDate")
      .resample("MS")
      .agg(
          Sales=("LineTotal", "sum"),
          Quantity=("Quantity", "sum"),
          Invoices=("InvoiceID", "nunique"),
          Customers=("CustomerID", "nunique"),
      )
      .reset_index()
)

monthly["Sales"] = monthly["Sales"].astype(float)
monthly["Quantity"] = monthly["Quantity"].astype(float)

# Optionally drop the latest (possibly incomplete) month
if exclude_last and len(monthly) > 1:
    monthly = monthly.iloc[:-1].reset_index(drop=True)

if len(monthly) < 4:
    st.warning(
        "At least 4 months of historical data are recommended "
        "for forecasting."
    )
    st.stop()

if len(monthly) < 24:
    st.info(
        "Fewer than 24 months of history: the seasonal terms are "
        "estimated from limited data, so treat the forecast with caution."
    )


# ---------------------------------------------------------------------
# FEATURE ENGINEERING — MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------------------
# Predictors:
#   trend  = months since the first month in the training data
#   sin12  = sin(2*pi*month/12)   } smooth yearly seasonality
#   cos12  = cos(2*pi*month/12)   }

FEATURES = ["trend", "sin12", "cos12"]


def make_features(dates, origin):

    dates = pd.to_datetime(pd.Series(dates))

    month = dates.dt.month

    trend = (
        (dates.dt.year - origin.year) * 12
        + (dates.dt.month - origin.month)
    ).astype(float)

    return pd.DataFrame({
        "trend": trend.values,
        "sin12": np.sin(2 * np.pi * month / 12).values,
        "cos12": np.cos(2 * np.pi * month / 12).values,
    })


def mape_fn(actual, pred):

    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)

    nonzero = actual != 0

    if nonzero.any():
        return np.mean(
            np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])
        ) * 100

    return np.nan


# ---------------------------------------------------------------------
# FORECAST MODEL — MULTIPLE LINEAR REGRESSION
# Y = b0 + b1*Trend + b2*sin12 + b3*cos12
# ---------------------------------------------------------------------

def forecast_series(history, periods, target):

    hist = history[["InvoiceDate", target]].copy()

    hist[target] = pd.to_numeric(hist[target], errors="coerce").fillna(0)

    origin = hist["InvoiceDate"].min()

    X = make_features(hist["InvoiceDate"], origin)
    y = hist[target].values

    model = LinearRegression()
    model.fit(X[FEATURES], y)

    future_dates = pd.date_range(
        hist["InvoiceDate"].max() + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    X_future = make_features(future_dates, origin)

    pred = model.predict(X_future[FEATURES])

    # Sales / Quantity cannot be negative
    pred = np.maximum(pred, 0)

    result = pd.DataFrame({
        "InvoiceDate": future_dates,
        "Predicted": pred,
    })

    return model, result, origin


# ---------------------------------------------------------------------
# SELECT TARGET
# ---------------------------------------------------------------------

target_column = "Sales" if metric == "Sales" else "Quantity"

model, forecast, origin = forecast_series(
    monthly,
    forecast_months,
    target_column,
)


# ---------------------------------------------------------------------
# MODEL QUALITY / HOLDOUT TEST
# Train on all but the last 3 months, test on those 3 months.
# Also compares against a naive baseline (mean of last 3 train months).
# ---------------------------------------------------------------------

def holdout_score(history, target):

    if len(history) < 8:
        return None

    train = history.iloc[:-3].copy()
    test = history.iloc[-3:].copy()

    train_origin = train["InvoiceDate"].min()

    X_train = make_features(train["InvoiceDate"], train_origin)
    X_test = make_features(test["InvoiceDate"], train_origin)

    y_train = train[target].values
    actual = test[target].values

    m = LinearRegression()
    m.fit(X_train[FEATURES], y_train)

    pred = np.maximum(m.predict(X_test[FEATURES]), 0)

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mape_fn(actual, pred)

    # Naive baseline: predict the average of the last 3 training months
    baseline_pred = np.repeat(y_train[-3:].mean(), len(actual))
    base_mape = mape_fn(actual, baseline_pred)

    return mae, rmse, mape, base_mape


scores = holdout_score(monthly, target_column)


# ---------------------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------------------

latest_month = monthly.iloc[-1]
previous_month = monthly.iloc[-2]

sales_growth = 0

if previous_month["Sales"] != 0:
    sales_growth = (
        (latest_month["Sales"] - previous_month["Sales"])
        / abs(previous_month["Sales"])
    ) * 100

forecast_total = forecast["Predicted"].sum()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Sales", f"{monthly['Sales'].sum():,.2f}")

with c2:
    st.metric(
        "Latest Month Sales",
        f"{latest_month['Sales']:,.2f}",
        f"{sales_growth:+.1f}%",
    )

with c3:
    st.metric("Historical Quantity", f"{monthly['Quantity'].sum():,.0f}")

with c4:
    label = "Forecast Sales" if metric == "Sales" else "Forecast Quantity"
    st.metric(label, f"{forecast_total:,.0f}")


# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🔮 Future Forecast",
    "📈 Prediction Analysis",
    "📋 Data",
])


# =====================================================================
# TAB 1 - DASHBOARD
# =====================================================================

with tab1:

    st.subheader("Monthly Sales Trend")

    fig_sales = px.line(
        monthly,
        x="InvoiceDate",
        y="Sales",
        markers=True,
        title="Historical Monthly Sales",
        labels={"InvoiceDate": "Month", "Sales": "Sales"},
    )

    fig_sales.update_layout(hovermode="x unified")

    st.plotly_chart(
        fig_sales,
        use_container_width=True,
        key="historical_sales_chart",
    )

    col1, col2 = st.columns(2)

    with col1:

        category = (
            df.groupby("Category", as_index=False)["LineTotal"]
            .sum()
            .sort_values("LineTotal", ascending=False)
        )

        fig_category = px.bar(
            category,
            x="Category",
            y="LineTotal",
            title="Sales by Category",
            labels={"LineTotal": "Sales"},
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True,
            key="category_sales_chart",
        )

    with col2:

        product = (
            df.groupby("ProductName", as_index=False)["LineTotal"]
            .sum()
            .sort_values("LineTotal", ascending=False)
            .head(10)
        )

        fig_product = px.bar(
            product,
            x="LineTotal",
            y="ProductName",
            orientation="h",
            title="Top 10 Products by Sales",
            labels={"LineTotal": "Sales", "ProductName": "Product"},
        )

        st.plotly_chart(
            fig_product,
            use_container_width=True,
            key="top_products_chart",
        )


# =====================================================================
# TAB 2 - FUTURE FORECAST
# =====================================================================

with tab2:

    st.subheader(f"🔮 Next {forecast_months} Months — {metric} Forecast")

    hist_plot = monthly[["InvoiceDate", target_column]].copy()
    hist_plot = hist_plot.rename(columns={target_column: "Value"})

    forecast_plot = forecast.rename(columns={"Predicted": "Value"})

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=hist_plot["InvoiceDate"],
            y=hist_plot["Value"],
            mode="lines+markers",
            name="Historical",
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=forecast_plot["InvoiceDate"],
            y=forecast_plot["Value"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash"),
        )
    )

    fig_forecast.update_layout(
        title=f"Historical vs Future {metric}",
        xaxis_title="Month",
        yaxis_title=metric,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True,
        key="future_forecast_chart",
    )

    st.subheader("Predicted Future Months")

    display_forecast = forecast.copy()

    display_forecast["Month"] = display_forecast["InvoiceDate"].dt.strftime(
        "%B %Y"
    )

    display_forecast[f"Predicted {metric}"] = display_forecast[
        "Predicted"
    ].round(2)

    st.dataframe(
        display_forecast[["Month", f"Predicted {metric}"]].reset_index(
            drop=True
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv = display_forecast[["Month", f"Predicted {metric}"]].to_csv(
        index=False
    )

    st.download_button(
        "⬇️ Download Forecast CSV",
        csv,
        file_name="future_sales_forecast.csv",
        mime="text/csv",
        key="download_forecast",
    )


# =====================================================================
# TAB 3 - PREDICTION ANALYSIS
# =====================================================================

with tab3:

    st.subheader("Prediction Model Analysis")

    if scores is not None:

        mae, rmse, mape, base_mape = scores

        a, b, c, d = st.columns(4)

        with a:
            st.metric("MAE", f"{mae:,.2f}")

        with b:
            st.metric("RMSE", f"{rmse:,.2f}")

        with c:
            st.metric(
                "MAPE (model)",
                "N/A" if np.isnan(mape) else f"{mape:.2f}%",
            )

        with d:
            st.metric(
                "MAPE (naive baseline)",
                "N/A" if np.isnan(base_mape) else f"{base_mape:.2f}%",
            )

        st.caption(
            "Metrics are calculated by training on all but the last 3 "
            "historical months and testing on those 3 months. The naive "
            "baseline predicts the average of the last 3 training months; "
            "if the model's MAPE is not lower than the baseline's, the "
            "model is not adding value. These are diagnostic measures and "
            "do not guarantee future accuracy."
        )

    else:
        st.info(
            "At least 8 historical months are needed for the "
            "holdout accuracy test."
        )

    # -------------------------------------------------------------
    # MODEL EQUATION — MULTIPLE LINEAR REGRESSION
    # -------------------------------------------------------------

    st.subheader("Multiple Linear Regression Equation")

    intercept = model.intercept_
    beta_trend, beta_sin, beta_cos = model.coef_

    st.write("The model uses the following equation:")

    st.latex(
        r"""
        \hat{Y} =
        \beta_0 +
        \beta_1(Trend) +
        \beta_2\sin\left(\frac{2\pi \cdot Month}{12}\right) +
        \beta_3\cos\left(\frac{2\pi \cdot Month}{12}\right)
        """
    )

    equation_text = (
        f"Predicted {metric} = "
        f"{intercept:.4f} + "
        f"({beta_trend:.4f} × Trend) + "
        f"({beta_sin:.4f} × sin12) + "
        f"({beta_cos:.4f} × cos12)"
    )

    st.code(equation_text)

    coefficient_table = pd.DataFrame({
        "Feature": [
            "Intercept",
            "Trend coefficient",
            "sin12 (seasonal)",
            "cos12 (seasonal)",
        ],
        "Coefficient": [
            intercept,
            beta_trend,
            beta_sin,
            beta_cos,
        ],
    })

    st.dataframe(
        coefficient_table,
        use_container_width=True,
        hide_index=True,
    )

    # -------------------------------------------------------------
    # ACTUAL VS MODELLED VALUES
    # -------------------------------------------------------------

    X_hist = make_features(monthly["InvoiceDate"], origin)

    fitted = np.maximum(model.predict(X_hist[FEATURES]), 0)

    prediction_check = pd.DataFrame({
        "InvoiceDate": monthly["InvoiceDate"],
        "Actual": monthly[target_column],
        "Model": fitted,
    })

    fig_model = go.Figure()

    fig_model.add_trace(
        go.Scatter(
            x=prediction_check["InvoiceDate"],
            y=prediction_check["Actual"],
            mode="lines+markers",
            name="Actual",
        )
    )

    fig_model.add_trace(
        go.Scatter(
            x=prediction_check["InvoiceDate"],
            y=prediction_check["Model"],
            mode="lines",
            name="Model",
        )
    )

    fig_model.update_layout(
        title=f"Actual vs Modelled Monthly {metric}",
        xaxis_title="Month",
        yaxis_title=metric,
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_model,
        use_container_width=True,
        key="actual_vs_model_chart",
    )

    # -------------------------------------------------------------
    # MONTHLY GROWTH
    # -------------------------------------------------------------

    st.subheader("Monthly Growth")

    growth = monthly.copy()

    growth["Growth %"] = growth[target_column].pct_change() * 100

    fig_growth = px.bar(
        growth,
        x="InvoiceDate",
        y="Growth %",
        title=f"Monthly {metric} Growth",
        labels={"InvoiceDate": "Month"},
    )

    st.plotly_chart(
        fig_growth,
        use_container_width=True,
        key="monthly_growth_chart",
    )


# =====================================================================
# TAB 4 - DATA
# =====================================================================

with tab4:

    st.subheader("Sales Data")
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,                      # required
    file_name="sales_forecast.csv",
    mime="text/csv",
)


