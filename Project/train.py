import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Paths
RAW_CSV = "data/train_data.csv"
PREPROCESSED_CSV = "data/train_data_preprocessed.csv"
PIPELINE_PKL = "models/data_preprocessing_pipeline.pkl"

# Expected feature columns (original numeric + one-hot encoded categorical)
features = [
    "Latitude",
    "Longitude",
    "Depth",
    "Magnitude",
    "Root Mean Square",
    "Magnitude Type_MD",
    "Magnitude Type_MH",
    "Magnitude Type_ML",
    "Magnitude Type_MS",
    "Magnitude Type_MW",
    "Magnitude Type_MWB",
    "Magnitude Type_MWC",
    "Magnitude Type_MWR",
    "Magnitude Type_MWW",
    "Status_Reviewed"
]

# Load (or create) preprocessed data with one-hot encoded columns
if os.path.exists(PREPROCESSED_CSV):
    df = pd.read_csv(PREPROCESSED_CSV)
else:
    # Read raw data
    df_raw = pd.read_csv(RAW_CSV, low_memory=False)

    # Require preprocessing pipeline
    if not os.path.exists(PIPELINE_PKL):
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at {PIPELINE_PKL}. Run data_preprocessing_pipeline.py first.")

    pipeline = joblib.load(PIPELINE_PKL)
    df = pipeline.transform(df_raw)
    # Save processed CSV so subsequent runs are fast
    os.makedirs(os.path.dirname(PREPROCESSED_CSV), exist_ok=True)
    df.to_csv(PREPROCESSED_CSV, index=False)

# Ensure all expected feature columns exist (fill missing one-hot columns with zeros)
for c in features:
    if c not in df.columns:
        df[c] = 0

# Prepare feature matrix and target vector
if 'Damage_Potential' not in df.columns:
    raise KeyError('Damage_Potential not found in preprocessed data. Run the preprocessing pipeline first.')

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
