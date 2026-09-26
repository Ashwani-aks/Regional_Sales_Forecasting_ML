import pandas as pd

df = pd.read_csv("data/sales_data.csv")
df["Date"] = pd.to_datetime(df["Date"]) # text to numeric date

# puts records with the same date, product, and region together
daily = (
    df.groupby(["Date", "Product ID", "Region"])
    .agg(                 # create 1 new row per category given
        Price=("Price", "mean"),
        Discount=("Discount", "mean"),
        Promotion=("Holiday/Promotion", "max"),
        Units_Sold=("Units Sold", "sum")
    )
)

all_dates = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
all_products = sorted(df["Product ID"].unique())
all_regions = sorted(df["Region"].unique())

complete_index = pd.MultiIndex.from_product( # Create every possible Date + Product + Region combination
    [all_dates, all_products, all_regions],
    names=["Date", "Product ID", "Region"]
)

daily = daily.reindex(complete_index).reset_index()    # if a combination missing it will make it new row 

#missing sales --> 0
daily["Units_Sold"] = daily["Units_Sold"].fillna(0)

#missing price ----> median price
daily["Price"] = daily["Price"].fillna(
    daily.groupby("Product ID")["Price"].transform("median")
)

#missing discount and promotion --> 0
daily["Discount"] = daily["Discount"].fillna(0)
daily["Promotion"] = daily["Promotion"].fillna(0)

# Assign the most common category to every product
product_categories = (
    df.groupby("Product ID")["Category"]
    .agg(lambda values: values.mode().iloc[0])
)

daily["Category"] = daily["Product ID"].map(product_categories)

# month from date
daily["Month"] = daily["Date"].dt.month

def get_season(month):
    if month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    elif month in [11, 12, 1, 2]:
        return "Winter"
    else:
        return "Autumn"

daily["Season"] = daily["Month"].apply(get_season)

# Sort records before calculating historical and future sales
daily = daily.sort_values(
    ["Product ID", "Region", "Date"]
).reset_index(drop=True)

groups = daily.groupby(["Product ID", "Region"])

# add previous 7 day sale and previous 30 day sale
daily["Previous_7_Day_Sales"] = groups["Units_Sold"].transform(
    lambda sales: sales.rolling(7).sum().shift(1)
)

daily["Previous_30_Day_Sales"] = groups["Units_Sold"].transform(
    lambda sales: sales.rolling(30).sum().shift(1)
)

# Future sales targets
daily["Next_7_Day_Sales"] = sum(
    groups["Units_Sold"].shift(-day)
    for day in range(1, 8)
)

daily["Next_30_Day_Sales"] = sum(
    groups["Units_Sold"].shift(-day)
    for day in range(1, 31)
)

daily = daily.dropna().reset_index(drop=True)

daily.to_csv("data/processed_sales_data.csv", index=False)

print("Data prepared successfully")
print("Processed shape:", daily.shape)

print("\nSample records:")
print(
    daily[
        [
            "Date",
            "Product ID",
            "Region",
            "Units_Sold",
            "Previous_7_Day_Sales",
            "Previous_30_Day_Sales",
            "Next_7_Day_Sales",
            "Next_30_Day_Sales"
        ]
    ].head(10)
)

# Confirm that dates are consecutive
sample_dates = daily[
    (daily["Product ID"] == "P0001") &
    (daily["Region"] == "East")
]["Date"].head(10)

print("\nConsecutive dates check:")
print(sample_dates.to_list())

print("\nMissing values:", daily.isnull().sum().sum())