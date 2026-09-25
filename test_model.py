import pandas as pd
import joblib

# Load prepared data and trained models
df = pd.read_csv("data/processed_sales_data.csv")

model_7 = joblib.load("models/sales_model_7_days.joblib")
model_30 = joblib.load("models/sales_model_30_days.joblib")

# Use one existing record as a test input
sample = df.iloc[[0]]

features = [
    "Product ID",
    "Category",
    "Region",
    "Price",
    "Discount",
    "Promotion",
    "Season",
    "Month",
    "Units_Sold",
    "Previous_7_Day_Sales",
    "Previous_30_Day_Sales"
]

X_sample = sample[features]

prediction_7 = model_7.predict(X_sample)[0]
prediction_30 = model_30.predict(X_sample)[0]

print("Test input:")
print(sample[["Date", "Product ID", "Region"]])

print("\nActual next 7-day sales:", round(sample["Next_7_Day_Sales"].iloc[0]))
print("Predicted next 7-day sales:", round(prediction_7))

print("\nActual next 30-day sales:", round(sample["Next_30_Day_Sales"].iloc[0]))
print("Predicted next 30-day sales:", round(prediction_30))