import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load preprocessed data with one-hot encoded columns
df = pd.read_csv("data/processed_trained_data.csv")

# Load processed data and get all feature columns automatically
df = pd.read_csv("data/processed_trained_data.csv")

# Select feature columns (all except Damage_Potential which is our target)
features = [col for col in df.columns if col != 'Damage_Potential']
print(f"Feature columns: {features}")
print(f"Target column: Damage_Potential")

# Prepare feature matrix and target vector
X = df[features]
y = df["Damage_Potential"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train LightGBM regressor
model = lgb.LGBMRegressor(
    n_estimators=600,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

# Save model and feature list
os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    "models/lgb_damage_model.pkl"
)
print("Saved model and feature list")