import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained models
model_7 = joblib.load("models/sales_model_7_days.joblib")
model_30 = joblib.load("models/sales_model_30_days.joblib")

st.title("Regional Sales Forecasting")

products = [f"P{i:04d}" for i in range(1, 21)]   #create list of product id from 1 to 20
regions = ["North", "South", "East", "West"]
categories = ["Groceries", "Toys", "Electronics", "Furniture", "Clothing"]
seasons = ["Winter", "Summer", "Monsoon", "Autumn"]


col1, col2 = st.columns(2)

with col1:
    product = st.selectbox("Select Product ID", products)
    category = st.selectbox("Select Product Category", categories)
    region = st.selectbox("Select Region", regions)

    price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=100.0
    )

    discount = st.number_input(
        "Discount Percentage",
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

with col2:
    promotion_text = st.selectbox(
        "Holiday / Promotion",
        ["No", "Yes"]
    )

    season = st.selectbox("Select Season", seasons)

    month = st.number_input(
        "Month Number",
        min_value=1,
        max_value=12,
        value=1
    )

    units_sold = st.number_input(
        "Today's Units Sold",
        min_value=0.0,
        value=100.0
    )

    previous_7_days = st.number_input(
        "Previous 7-Day Sales",
        min_value=0.0,
        value=700.0
    )

    previous_30_days = st.number_input(
        "Previous 30-Day Sales",
        min_value=0.0,
        value=3000.0
    )

promotion = 1 if promotion_text == "Yes" else 0

forecast_period = st.radio(
    "Select Forecast Period",
    ["Next 7 Days", "Next 30 Days"],
    horizontal=True
)

if st.button("Predict Sales"):

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

    if forecast_period == "Next 7 Days":
        prediction = model_7.predict(input_data)[0]
        forecast_name = "Next 7 Days"
    else:
        prediction = model_30.predict(input_data)[0]
        forecast_name = "Next 30 Days"

    prediction = max(0, round(prediction))
    recommended_stock = round(prediction * 1.10)

    st.subheader("Prediction Result")

    result1, result2 = st.columns(2)

    with result1:
        st.write("Product ID:", product)
        st.write("Region:", region)
        st.write("Forecast Period:", forecast_name)

    with result2:
        st.write("Predicted Sales:", prediction, "units")
        st.write("Recommended Stock:", recommended_stock, "units")

    
    labels = ["Predicted Sales", "Recommended Stock"]
    values = [prediction, recommended_stock]

    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=["blue", "green"])

    ax.set_title("Sales Forecast Result")
    ax.set_ylabel("Units")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(round(height)),
            ha="center",
            va="bottom"
        )

    st.pyplot(fig)