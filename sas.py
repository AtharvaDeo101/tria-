import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("sales_data.csv")

# Encode categorical variables
encoder = LabelEncoder()
df["shop_id"] = encoder.fit_transform(df["shop_id"])
df["Product_ID"] = encoder.fit_transform(df["Product_ID"])

# Select features and target
X = df[["shop_id", "Product_ID", "Historical_Sales", "Promotion", "Day_of_Week", "Month"]]
y = df["Demand"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Model Performance: MAE={mae:.2f}, R²={r2:.2f}")

# Plot the most active shop (highest sales)
active_shop = df.groupby("shop_id")["Historical_Sales"].sum().reset_index()
active_shop = active_shop.sort_values(by="Historical_Sales", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=active_shop["shop_id"], y=active_shop["Historical_Sales"], palette="viridis")
plt.xlabel("Shop ID")
plt.ylabel("Total Sales")
plt.title("Most Active Shops Based on Sales")
plt.xticks(rotation=45)
plt.show()
