import os
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="ImpactSense", layout="wide")

# Custom CSS for gradient background and enhanced UI
st.markdown("""
<style>
    /* Main background gradient with smooth wave animation */
    .stApp {
        background: linear-gradient(45deg, #00c851, #2980b9, #1e88e5, #00c851, #2980b9);
        background-size: 400% 400%;
        background-attachment: fixed;
        animation: liquidFlow 12s ease-in-out infinite;
    }
    
    @keyframes liquidFlow {
        0% {
            background-position: 0% 50%;
            background: radial-gradient(circle at 20% 80%, #00c851 0%, transparent 50%), 
                        radial-gradient(circle at 80% 20%, #2980b9 0%, transparent 50%), 
                        radial-gradient(circle at 40% 40%, #1e88e5 0%, transparent 50%), 
                        linear-gradient(45deg, #16a085, #3498db);
        }
        25% {
            background-position: 100% 50%;
            background: radial-gradient(circle at 60% 30%, #2980b9 0%, transparent 50%), 
                        radial-gradient(circle at 30% 70%, #1e88e5 0%, transparent 50%), 
                        radial-gradient(circle at 80% 80%, #16a085 0%, transparent 50%), 
                        linear-gradient(45deg, #3498db, #00c851);
        }
        50% {
            background-position: 100% 100%;
            background: radial-gradient(circle at 80% 20%, #1e88e5 0%, transparent 50%), 
                        radial-gradient(circle at 20% 60%, #16a085 0%, transparent 50%), 
                        radial-gradient(circle at 60% 90%, #3498db 0%, transparent 50%), 
                        linear-gradient(45deg, #00c851, #2980b9);
        }
        75% {
            background-position: 0% 100%;
            background: radial-gradient(circle at 40% 70%, #16a085 0%, transparent 50%), 
                        radial-gradient(circle at 70% 40%, #3498db 0%, transparent 50%), 
                        radial-gradient(circle at 10% 10%, #00c851 0%, transparent 50%), 
                        linear-gradient(45deg, #2980b9, #1e88e5);
        }
        100% {
            background-position: 0% 50%;
            background: radial-gradient(circle at 20% 80%, #00c851 0%, transparent 50%), 
                        radial-gradient(circle at 80% 20%, #2980b9 0%, transparent 50%), 
                        radial-gradient(circle at 40% 40%, #1e88e5 0%, transparent 50%), 
                        linear-gradient(45deg, #16a085, #3498db);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(0, 200, 81, 0.15) 0%, rgba(41, 128, 185, 0.25) 50%, rgba(30, 136, 229, 0.15) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced sidebar elements */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(25px);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 1.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    /* Floating card effect for sections */
    .stContainer > div {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .stContainer > div:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Headers styling */
    .stMarkdown h1 {
        color: #2c3e50;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        background: linear-gradient(90deg, #00c851, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    
    /* Main title with animated color-changing glow effect */
    .main-title {
        font-weight: bold;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 20px;
        border: 3px solid rgba(255, 255, 255, 0.8);
        animation: colorGlow 6s ease-in-out infinite;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.3), inset 0 0 20px rgba(255, 255, 255, 0.2);
    }
    
    @keyframes colorGlow {
        0% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(0, 200, 81, 0.6);
            box-shadow: 0 0 20px rgba(0, 200, 81, 0.4), inset 0 0 20px rgba(0, 200, 81, 0.1);
        }
        16.66% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(41, 128, 185, 0.6);
            box-shadow: 0 0 20px rgba(41, 128, 185, 0.4), inset 0 0 20px rgba(41, 128, 185, 0.1);
        }
        33.33% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(241, 196, 15, 0.6);
            box-shadow: 0 0 20px rgba(241, 196, 15, 0.4), inset 0 0 20px rgba(241, 196, 15, 0.1);
        }
        50% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(230, 126, 34, 0.6);
            box-shadow: 0 0 20px rgba(230, 126, 34, 0.4), inset 0 0 20px rgba(230, 126, 34, 0.1);
        }
        66.66% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(155, 89, 182, 0.6);
            box-shadow: 0 0 20px rgba(155, 89, 182, 0.4), inset 0 0 20px rgba(155, 89, 182, 0.1);
        }
        83.33% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(52, 152, 219, 0.6);
            box-shadow: 0 0 20px rgba(52, 152, 219, 0.4), inset 0 0 20px rgba(52, 152, 219, 0.1);
        }
        100% {
            color: #000000 !important;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            border-color: rgba(0, 200, 81, 0.6);
            box-shadow: 0 0 20px rgba(0, 200, 81, 0.4), inset 0 0 20px rgba(0, 200, 81, 0.1);
        }
    }
    
    /* Additional glow effect for the title */
    .main-title::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        border-radius: 22px;
        animation: shimmer 3s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        50% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    .stMarkdown h2 {
        color: #34495e;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .stMarkdown h3 {
        color: #2980b9;
        font-weight: 600;
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(0, 200, 81, 0.15), rgba(30, 136, 229, 0.15));
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced metrics display */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.6));
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #00c851, #2980b9);
        color: black;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 2px solid rgba(0, 200, 81, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: rgba(0, 200, 81, 0.6);
        box-shadow: 0 5px 15px rgba(0, 200, 81, 0.2);
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #00c851, #2980b9);
        border-radius: 10px;
    }
    
    /* Enhanced input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: rgba(0, 200, 81, 0.6);
        box-shadow: 0 0 20px rgba(0, 200, 81, 0.3);
        background: rgba(255, 255, 255, 0.95);
    }
    
    /* Multiselect enhancement */
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        border: 2px solid rgba(0, 200, 81, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Data frame styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(15px);
    }
    
    /* Chart containers */
    .stPlotlyChart, .stPyplot {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(20px);
        transition: all 0.3s ease;
    }
    
    .stPlotlyChart:hover, .stPyplot:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.15);
    }
    
    /* Subheader styling */
    .stMarkdown h2, .stMarkdown h3 {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Warning and info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Warning text styling for better visibility */
    .stAlert > div {
        color: black !important;
        font-weight: 500;
    }
    
    .stAlert [data-testid="alertText"] {
        color: black !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(0, 200, 81, 0.15), rgba(30, 136, 229, 0.15));
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(0, 200, 81, 0.25), rgba(30, 136, 229, 0.25));
        transform: translateX(5px);
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        margin-top: 0.5rem;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 0.8rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, rgba(0, 200, 81, 0.1), rgba(30, 136, 229, 0.1));
        border-radius: 15px;
        margin: 0.3rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(90deg, rgba(0, 200, 81, 0.3), rgba(30, 136, 229, 0.3));
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    .stSpinner {
        animation: pulse 2s infinite;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00c851, #2980b9);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #00a844, #236da3);
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
def create_risk_category(score):
    if score < 4.0:
        return 0
    if score < 6.0:
        return 1
    return 2

def create_urban_risk(lat, lon, score):
    return score * (1 + (abs(lat) + abs(lon)) / 360)

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    fig, _ = plt.subplots(figsize=(8,5))
    shap.summary_plot(sv, X, show=False)
    return fig

# Preprocessing helper functions & classes
class DropNaNRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.dropna(axis=0)

class DamagePotentialCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        mag = X['Magnitude']
        depth = X['Depth']
        damage_potential = 0.6 * mag + 0.2 * (700 - depth) / 700 * 10
        X = X.copy()
        X['Damage_Potential'] = damage_potential
        return X

def select_features(X):
    return X[['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Type', 'Magnitude Type', 'Status', 'Root Mean Square']]

def impute_root_mean_square(df):
    known_df = df[df['Root Mean Square'].notna()]
    unknown_df = df[df['Root Mean Square'].isna()]

    if unknown_df.empty:
        return df

    features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
    X_train = known_df[features]
    y_train = known_df['Root Mean Square']
    X_pred = unknown_df[features]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predicted = model.predict(X_pred)
    df.loc[df['Root Mean Square'].isna(), 'Root Mean Square'] = predicted
    return df

def filter_earthquake(X):
    return X[X['Type'] == 'Earthquake']

def drop_type(X):
    return X.drop('Type', axis=1)

def encode_categoricals(X):
    categorical_cols = ['Magnitude Type', 'Status']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X = X.drop(columns=categorical_cols).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    return pd.concat([X, encoded_df], axis=1)

def transform_input(input_dict, feat_list, df_raw):
    X = pd.DataFrame([input_dict])

    # Impute Root Mean Square if missing
    if pd.isna(X.loc[0, 'Root Mean Square']):
        # Use median from raw data as fallback
        median_rms = df_raw['Root Mean Square'].median()
        X.loc[0, 'Root Mean Square'] = median_rms

    # Skip damage potential creation as it's not used in features
    
    # Skip type filtering since we only process earthquakes
    if X.loc[0, 'Type'] != 'Earthquake':
        return pd.DataFrame([[0] * len(feat_list)], columns=feat_list)

    # One-hot encode categorical columns to match training features
    categorical_mappings = {
        'Magnitude Type': ['MD', 'MH', 'ML', 'MS', 'MW', 'MWB', 'MWC', 'MWR', 'MWW', 'Unknown'],
        'Status': ['Reviewed']
    }
    
    # Create one-hot encoded features
    for cat_col, categories in categorical_mappings.items():
        cat_value = X.loc[0, cat_col]
        for category in categories:
            col_name = f"{cat_col}_{category}"
            if col_name in feat_list:
                X[col_name] = 1 if cat_value == category else 0
    
    # Drop original categorical columns
    X = X.drop(columns=['Type', 'Magnitude Type', 'Status'])
    
    # Ensure all expected features are present
    for col in feat_list:
        if col not in X.columns:
            X[col] = 0

    # Return features in correct order
    X = X[feat_list]
    
    return X

# Sidebar interface
with st.sidebar:
    uploaded_file = st.file_uploader("Upload earthquake CSV file", type=["csv"])
    page = st.radio("Page", ["📊 Data", "🔮 Predict", "🗺️ Map", "ℹ️ About"], index=0)

# Fixed paths for preprocessing pipeline and model
preprocessing_path = "models/data_preprocessing_pipeline.pkl"
model_path = "models/lgb_damage_model.pkl"

if uploaded_file is None:
    st.warning("Please upload an earthquake CSV file to proceed.")
    st.stop()

# Load raw data from uploaded file
df_raw = pd.read_csv(uploaded_file).dropna(subset=["Latitude", "Longitude", "Depth", "Magnitude"]).copy()

# Use direct preprocessing instead of pipeline to avoid pickle issues
def preprocess_for_display(df_raw):
    """Simplified preprocessing for display purposes"""
    # Select relevant features
    df = df_raw[['Latitude', 'Longitude', 'Type', 'Depth', 'Magnitude', 
                 'Magnitude Type', 'Root Mean Square', 'Status']].copy()
    
    # Drop rows with missing essential data
    df = df.dropna(subset=['Latitude', 'Longitude', 'Depth', 'Magnitude'])
    
    # Simple imputation for Root Mean Square (use median)
    if df['Root Mean Square'].isna().any():
        median_rms = df['Root Mean Square'].median()
        df['Root Mean Square'] = df['Root Mean Square'].fillna(median_rms)
    
    # Filter only earthquakes
    df = df[df['Type'] == 'Earthquake'].copy()
    
    # Create Damage_Potential feature
    mag = df['Magnitude']
    depth = df['Depth']
    damage_potential = 0.6 * mag + 0.2 * (700 - depth) / 700 * 10
    df['Damage_Potential'] = damage_potential
    
    return df

# Preprocess raw data for display
df_processed = preprocess_for_display(df_raw)

# Get only numeric features for display (exclude categorical columns)
numeric_features = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Root Mean Square']
features = [col for col in numeric_features if col in df_processed.columns]

# Load model and features
if not os.path.exists(model_path):
    st.error("Model missing on server.")
    st.stop()
saved = joblib.load(model_path)
model, feat_list = saved["model"], saved["features"]

# Add Risk Category only for display (no Urban Risk column)
if 'Damage_Potential' in df_processed.columns:
    df_processed["Risk_Category"] = df_processed["Damage_Potential"].apply(create_risk_category)
else:
    df_processed["Risk_Category"] = None

# Pages
if page == "📊 Data":
    st.markdown('<h1 class="main-title">🌏 ImpactSense – Earthquake Impact Prediction & Risk Visualization</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **ImpactSense**, a comprehensive project designed to predict and visualize the impact of earthquakes.
    
    This application leverages machine learning models to estimate:
    
    - **Damage Potential**: a severity score based on earthquake magnitude and depth.
    - **Risk Category**: classification of earthquakes into low, moderate, or high risk.
    - **Urban Risk Score**: an adjusted damage potential factoring in geographical location as a proxy for population exposure.
    
    Use the navigation panel to explore earthquake data, make predictions, and visualize risk maps.
    """)
    st.subheader("Dataset Overview")
    # Show overview for numeric columns only
    numeric_display_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_display_cols:
        st.write("**Numeric Features Summary:**")
        st.write(df_processed[numeric_display_cols].describe())
    
    # Show categorical columns info
    categorical_display_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if categorical_display_cols:
        st.write("**Categorical Features:**")
        for col in categorical_display_cols:
            unique_vals = df_processed[col].nunique()
            st.write(f"- **{col}**: {unique_vals} unique values")
            if unique_vals <= 10:  # Show values if not too many
                st.write(f"  Values: {', '.join(map(str, df_processed[col].unique()[:10]))}")
    
    st.write(f"**Total Records:** {len(df_processed)}")
    st.write(f"**Total Features:** {len(df_processed.columns)}")

    st.markdown("### Distributions")
    # Separate numeric and categorical columns for different visualizations
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    all_display_cols = numeric_cols + categorical_cols
    cols = st.multiselect("Select columns", all_display_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
    
    for c in cols:
        fig, ax = plt.subplots()
        if c in numeric_cols:
            # Histogram for numeric columns
            ax.hist(df_processed[c].dropna(), bins=30, color="skyblue", edgecolor="black")
            ax.set_title(f"{c} Distribution")
            ax.set_xlabel(c)
            ax.set_ylabel("Frequency")
        else:
            # Bar plot for categorical columns
            value_counts = df_processed[c].value_counts()
            ax.bar(range(len(value_counts)), value_counts.values, color="lightcoral", edgecolor="black")
            ax.set_title(f"{c} Distribution")
            ax.set_xlabel(c)
            ax.set_ylabel("Count")
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        st.pyplot(fig)

    st.markdown("### Correlation Matrix")
    # Only use numeric columns for correlation matrix
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr = df_processed[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
          corr,
          annot=True,
          fmt=".2f",
          cmap="vlag",
          ax=ax,
          annot_kws={"size": 10},
          xticklabels=True,
          yticklabels=True
        )
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()

        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation matrix.")




elif page == "🔮 Predict":
    st.subheader("Single Prediction")

    base_numeric = ["Latitude", "Longitude", "Depth", "Magnitude", "Root Mean Square"]
    numeric_vals = {c: st.slider(c, float(df_raw[c].min()), float(df_raw[c].max()), float(df_raw[c].median())) for c in base_numeric}

    # type_options = df_raw['Type'].dropna().unique().tolist() if 'Type' in df_raw.columns else []
    magnitude_type_options = df_raw['Magnitude Type'].dropna().unique().tolist() if 'Magnitude Type' in df_raw.columns else []
    status_options = df_raw['Status'].dropna().unique().tolist() if 'Status' in df_raw.columns else []

    type_val = 'Earthquake' # Fixed as only 'Earthquake' is processed
    magnitude_type_val = st.selectbox('Magnitude Type', options=magnitude_type_options if magnitude_type_options else ['ML'])
    status_val = st.selectbox('Status', options=status_options if status_options else ['Reviewed'])

    input_vals = numeric_vals.copy()
    input_vals['Type'] = type_val
    input_vals['Magnitude Type'] = magnitude_type_val
    input_vals['Status'] = status_val

    if st.button("Predict"):
        X_input = transform_input(input_vals, feat_list, df_raw)

        dp = model.predict(X_input)[0]
        rc = create_risk_category(dp)
        ur = create_urban_risk(input_vals["Latitude"], input_vals["Longitude"], dp)

        st.metric("Damage Potential", f"{dp:.2f}")
        st.metric("Risk Category", ["Low", "Moderate", "High"][rc])
        st.metric("Urban Risk Score", f"{ur:.2f}")

        with st.expander("🔍 SHAP Explainability"):
            st.pyplot(shap_explain(model, X_input))

elif page == "🗺️ Map":
    st.subheader("Risk Map")
    df_map = df_processed.copy()
    if 'Damage_Potential' in df_map.columns:
        df_map["Risk_Label"] = df_map.Risk_Category.map({0: "Low", 1: "Moderate", 2: "High"})
        
        # Use only valid numeric columns for hover data
        numeric_features_for_map = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Root Mean Square']
        valid_hover_features = [col for col in numeric_features_for_map if col in df_map.columns]
        
        fig = px.scatter_mapbox(
            df_map, lat="Latitude", lon="Longitude",
            color="Risk_Label", size="Damage_Potential", size_max=15,
            zoom=1, mapbox_style="carto-positron",
            hover_data=valid_hover_features
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Sample Data")
        display_columns = valid_hover_features + ["Damage_Potential", "Risk_Label"]
        st.dataframe(df_map[display_columns].head(10))
    else:
        st.warning("Damage_Potential not found in processed data, cannot display map.")

else:
    st.subheader("About ImpactSense")
    st.markdown("""
### What is Damage Potential?
A numeric score estimating the earthquake's potential to cause destruction using magnitude and depth — higher means more damage expected.

### What is Risk Category?
A qualitative classification into:
- **Low**: minimal damage potential
- **Moderate**: potential for noticeable damage
- **High**: severe damage likely

### What is Urban Risk Score?
An adjusted damage potential factoring in location geography as a proxy for population density and infrastructure, highlighting areas where impact could affect more people.

This project helps visualize and predict earthquake impact to aid decision-making in disaster management and urban planning.
""")