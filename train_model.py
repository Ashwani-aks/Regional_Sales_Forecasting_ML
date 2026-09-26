import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load processed data
df = pd.read_csv("data/processed_sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"])


# Input features
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

categorical_features = [
    "Product ID",
    "Category",
    "Region",
    "Season"
]

numeric_features = [
    "Price",
    "Discount",
    "Promotion",
    "Month",
    "Units_Sold",
    "Previous_7_Day_Sales",
    "Previous_30_Day_Sales"
]


# get unique dates and split into train and test sets
unique_dates = sorted(df["Date"].unique())
split_position = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_position]

train_df = df[df["Date"] < split_date]
test_df = df[df["Date"] >= split_date]

X_train = train_df[features]
X_test = test_df[features]


def create_model():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numbers",
                "passthrough",
                numeric_features
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=7,
                    min_samples_leaf=15,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


def train_and_evaluate(target, model_path):
    y_train = train_df[target]
    y_test = test_df[target]

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, model_path)

    print(f"\nResults for {target}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Saved: {model_path}")

#print number of training and testing rows and the split date
print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Split date:", pd.Timestamp(split_date).date())

train_and_evaluate(
    "Next_7_Day_Sales",
    "models/sales_model_7_days.joblib"
)

train_and_evaluate(
    "Next_30_Day_Sales",
    "models/sales_model_30_days.joblib"
)

print("\nBoth models trained successfully.")