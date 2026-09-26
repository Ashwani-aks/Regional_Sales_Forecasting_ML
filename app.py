import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px


# webpage layout
st.set_page_config(
    page_title="Regional Sales Forecasting",
    page_icon="📈",
    layout="wide"
)


# dashboard styling css
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0;
    }

    .sub-title {
        font-size: 17px;
        color: #627d98;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 750;
        color: #102a43;
        margin: 8px 0 12px 0;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 14px;
        border-radius: 14px;
        border: 1px solid #e1e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 15px;
        font-weight: 700;
        color: #334e68;
    }

    div[data-testid="stMetricValue"] {
        font-size: 23px;
        font-weight: 800;
        color: #1d4ed8;
    }

    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e1e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        color: #243b53;
        line-height: 1.7;
    }

    .stButton button {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-size: 18px;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 11px;
    }

    label {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #243b53 !important;
    }
</style>
""", unsafe_allow_html=True)


# Load both saved ML models only once
@st.cache_resource
def load_models():
    seven_day_model = joblib.load("models/sales_model_7_days.joblib")
    thirty_day_model = joblib.load("models/sales_model_30_days.joblib")
    return seven_day_model, thirty_day_model


model_7, model_30 = load_models()


# Main dashboard heading
st.markdown(
    '<div class="main-title">📈 Regional Sales Forecasting Dashboard</div>',
    unsafe_allow_html=True
)
# explaination
st.markdown(
    '<div class="sub-title">Forecast demand, plan stock, and understand possible risk before making inventory decisions.</div>',
    unsafe_allow_html=True
)


# dropdown options
products = [f"P{i:04d}" for i in range(1, 21)]
regions = ["North", "South", "East", "West"]
categories = ["Groceries", "Toys", "Electronics", "Furniture", "Clothing"]
seasons = ["Winter", "Summer", "Monsoon", "Autumn"]


# do not make predictions until the button is clicked
with st.form("sales_forecast_form"):

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        st.markdown(
            '<div class="section-title">🛍️ Product Information</div>',
            unsafe_allow_html=True
        )
        # product info input
        product = st.selectbox("Select Product ID", products)
        category = st.selectbox("Select Product Category", categories)
        region = st.selectbox("Select Sales Region", regions)

        price = st.number_input(
            "Selling Price Per Unit (₹)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        # Cost is used only for calculating risk, not for ML prediction
        unit_cost = st.number_input(
            "Product Cost Per Unit (₹)",
            min_value=0.0,
            value=60.0,
            step=10.0,
            help="Used only to calculate possible profit loss and blocked inventory value."
        )

        discount = st.slider(
            "Discount Offered (%)",
            min_value=0,
            max_value=100,
            value=0
        )

    with input_col2:
        st.markdown(
            '<div class="section-title">📊 Sales & Market Information</div>',
            unsafe_allow_html=True
        )

        promotion_text = st.selectbox(
            "Holiday / Promotion Active?",
            ["No", "Yes"]
        )

        season = st.selectbox("Current Season", seasons)

        month = st.select_slider(
            "Current Month",
            options=list(range(1, 13)),
            value=1
        )

        units_sold = st.number_input(
            "Today's Units Sold",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

        previous_7_days = st.number_input(
            "Total Sales in Previous 7 Days",
            min_value=0.0,
            value=700.0,
            step=10.0
        )

        previous_30_days = st.number_input(
            "Total Sales in Previous 30 Days",
            min_value=0.0,
            value=3000.0,
            step=10.0
        )

        # uncertinity from 5 to 50 %
        trend_shift = st.slider(
            "Demand Uncertainty / Trend Shift (%)",
            min_value=5,
            max_value=50,
            value=20,
            help="Shows what can happen if actual demand becomes lower or higher than the ML forecast."
        )

    st.divider()

    forecast_period = st.radio(
        "Choose Prediction Period",
        ["Next 7 Days", "Next 30 Days"],
        horizontal=True
    )

    submitted = st.form_submit_button(
        "🔮 Generate Sales Forecast",
        use_container_width=True
    )


# Prediction section
if submitted:

    promotion = 1 if promotion_text == "Yes" else 0

    # These names must exactly match the columns used when your model was trained
    input_data = pd.DataFrame([{
        "Product ID": product,
        "Category": category,
        "Region": region,
        "Price": price,
        "Discount": discount,
        "Promotion": promotion,
        "Season": season,
        "Month": month,
        "Units_Sold": units_sold,
        "Previous_7_Day_Sales": previous_7_days,
        "Previous_30_Day_Sales": previous_30_days
    }])

    # Use the correct trained model
    if forecast_period == "Next 7 Days":
        prediction = model_7.predict(input_data)[0]
    else:
        prediction = model_30.predict(input_data)[0]

    prediction = max(0, round(prediction))

    # We keep 10% extra units as safety stock
    recommended_stock = round(prediction * 1.10)
    safety_stock = recommended_stock - prediction

    # Risk calculations if actual demand changes unexpectedly
    low_demand = round(prediction * (1 - trend_shift / 100))
    high_demand = round(prediction * (1 + trend_shift / 100))

    possible_shortage = max(0, high_demand - recommended_stock)
    possible_lost_revenue = possible_shortage * price
    possible_lost_profit = possible_shortage * max(0, price - unit_cost)

    excess_stock = max(0, recommended_stock - low_demand)
    blocked_inventory_value = excess_stock * unit_cost

    st.markdown("---")
    st.markdown(
        '<div class="section-title">✨ Sales Forecast Result</div>',
        unsafe_allow_html=True
    )

    # result metric cards
    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Product", product)
    metric2.metric("Region", region)
    metric3.metric("Predicted Sales", f"{prediction:,} units")
    metric4.metric(
        "Recommended Stock",
        f"{recommended_stock:,} units",
        delta=f"+{safety_stock:,} safety stock"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Useful interactive charts
    graph1, graph2 = st.columns(2)

    with graph1:
        donut_chart = go.Figure(data=[
            go.Pie(
                labels=["Expected Sales", "Safety Stock"],
                values=[prediction, safety_stock],
                hole=0.62,
                marker_colors=["#2563eb", "#86efac"],
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Units: %{value:,}<extra></extra>"
            )
        ])

        donut_chart.update_layout(
            title="Inventory Planning Breakdown",
            annotations=[dict(
                text=f"<b>{recommended_stock:,}</b><br>Units to Keep",
                x=0.5,
                y=0.5,
                font_size=18,
                showarrow=False
            )],
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=60, b=10, l=10, r=10)
        )

        st.plotly_chart(donut_chart, use_container_width=True)

    with graph2:
        sales_df = pd.DataFrame({
            "Period": [
                "Today",
                "Previous 7 Days",
                "Previous 30 Days",
                forecast_period
            ],
            "Sales Units": [
                units_sold,
                previous_7_days,
                previous_30_days,
                prediction
            ],
            "Type": ["Current", "History", "History", "Forecast"]
        })

        comparison_chart = px.bar(
            sales_df,
            x="Period",
            y="Sales Units",
            color="Type",
            text="Sales Units",
            color_discrete_map={
                "Current": "#f59e0b",
                "History": "#94a3b8",
                "Forecast": "#2563eb"
            }
        )

        comparison_chart.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Sales: %{y:,} units<extra></extra>"
        )

        comparison_chart.update_layout(
            title="Sales History vs Forecast",
            height=360,
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="",
            yaxis_title="Sales Units",
            margin=dict(t=60, b=10, l=10, r=10)
        )

        st.plotly_chart(comparison_chart, use_container_width=True)

    # No gauge graph here, so the dashboard stays compact
    summary_col, risk_col = st.columns([1.15, 1])

    with summary_col:
        st.markdown(f"""
        <div class="info-card">
            <h3>📋 Smart Inventory Plan</h3>
            <b>Product:</b> {product}<br>
            <b>Category:</b> {category}<br>
            <b>Region:</b> {region}<br>
            <b>Forecast duration:</b> {forecast_period}<br>
            <b>Promotion active:</b> {promotion_text}<br>
            <b>Expected demand:</b> {prediction:,} units<br>
            <b>Safety inventory:</b> {safety_stock:,} units<br>
            <b>Total stock to keep:</b> {recommended_stock:,} units<br><br>
            ✅ Keep <b>{recommended_stock:,} units</b> ready to reduce stock shortage risk.
        </div>
        """, unsafe_allow_html=True)

    with risk_col:
        st.markdown("### ⚠️ Risk Summary")
        st.caption(f"Demand uncertainty: ±{trend_shift}%")
        st.divider()

        high_col, low_col = st.columns(2)

        with high_col:
            st.markdown("#### 📈 High demand")
            st.metric("Demand", f"{high_demand:,}")
            st.metric("Shortage", f"{possible_shortage:,}")
            st.markdown(f"🔴 **Revenue risk:** ₹{possible_lost_revenue:,.0f}")
            st.markdown(f"🔴 **Profit risk:** ₹{possible_lost_profit:,.0f}")

        with low_col:
            st.markdown("#### 📉 Low demand")
            st.metric("Demand", f"{low_demand:,}")
            st.metric("Extra stock", f"{excess_stock:,}")
            st.markdown(f"🟠 **Blocked money:** ₹{blocked_inventory_value:,.0f}")

    st.success(
        f"Forecast generated successfully for {product} in the {region} region."
    )