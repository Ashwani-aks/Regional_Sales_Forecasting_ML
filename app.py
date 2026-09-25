import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Regional Sales Forecasting",
    page_icon="📈",
    layout="centered"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_sales_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_models():
    model_7 = joblib.load("models/sales_model_7_days.joblib")
    model_30 = joblib.load("models/sales_model_30_days.joblib")
    return model_7, model_30

df = load_data()
model_7, model_30 = load_models()

st.title("📈 Regional Product Sales Forecasting")
st.write(
    "Predict the next 7-day or 30-day product sales for a selected region."
)

products = sorted(df["Product ID"].unique())
regions = sorted(df["Region"].unique())

col1, col2 = st.columns(2)

with col1:
    selected_product = st.selectbox("Select Product", products)

with col2:
    selected_region = st.selectbox("Select Region", regions)

forecast_period = st.radio(
    "Forecast Period",
    ["Next 7 Days", "Next 30 Days"],
    horizontal=True
)

# Get the latest historical record for selected product and region
filtered_data = df[
    (df["Product ID"] == selected_product)
    & (df["Region"] == selected_region)
].sort_values("Date")

latest_record = filtered_data.iloc[-1]

st.subheader("Sales Conditions")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(
        "Price",
        min_value=0.0,
        value=float(latest_record["Price"])
    )

    discount = st.number_input(
        "Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(latest_record["Discount"])
    )

with col2:
    season = st.selectbox(
        "Season",
        ["Winter", "Summer", "Monsoon", "Autumn"],
        index=["Winter", "Summer", "Monsoon", "Autumn"].index(
            latest_record["Season"]
        )
    )

    promotion = st.selectbox(
        "Holiday / Promotion",
        [0, 1],
        format_func=lambda value: "Yes" if value == 1 else "No",
        index=int(latest_record["Promotion"])
    )

if st.button("Predict Sales", type="primary"):

    input_data = pd.DataFrame([{
        "Product ID": selected_product,
        "Category": latest_record["Category"],
        "Region": selected_region,
        "Price": price,
        "Discount": discount,
        "Promotion": promotion,
        "Season": season,
        "Month": latest_record["Month"],
        "Units_Sold": latest_record["Units_Sold"],
        "Previous_7_Day_Sales": latest_record["Previous_7_Day_Sales"],
        "Previous_30_Day_Sales": latest_record["Previous_30_Day_Sales"]
    }])

    if forecast_period == "Next 7 Days":
        predicted_sales = model_7.predict(input_data)[0]
        period_name = "next 7 days"
    else:
        predicted_sales = model_30.predict(input_data)[0]
        period_name = "next 30 days"

    predicted_sales = max(0, round(predicted_sales))
    recommended_stock = round(predicted_sales * 1.10)

    if predicted_sales < 1500:
        demand = "Low"
    elif predicted_sales < 4000:
        demand = "Medium"
    else:
        demand = "High"

    st.divider()
    st.subheader("Forecast Result")

    result1, result2, result3 = st.columns(3)

    result1.metric("Predicted Sales", f"{predicted_sales} units")
    result2.metric("Demand Level", demand)
    result3.metric("Recommended Stock", f"{recommended_stock} units")

    st.info(
        f"For **{selected_product}** in the **{selected_region}** region, "
        f"estimated sales for the **{period_name}** are **{predicted_sales} units**."
    )

    chart_data = pd.DataFrame(
        {
            "Type": ["Predicted Sales", "Recommended Stock"],
            "Units": [predicted_sales, recommended_stock]
        }
    ).set_index("Type")

    st.bar_chart(chart_data)

st.divider()
st.caption(
    "This project uses a Random Forest model trained on historical retail sales data."
)