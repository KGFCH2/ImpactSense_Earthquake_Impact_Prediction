# ImpactSense - Earthquake Impact Prediction

A machine learning project that predicts earthquake impact and visualizes risk assessment using LightGBM and Streamlit.

## Project Structure

```
Project/
├── README.md                      # 📖 This documentation file
├── app.py                         # 🌐 Streamlit web application
├── data_preprocessing_pipeline.py # 🔧 Data preprocessing pipeline
├── train.py                      # 🤖 Model training script
├── requirements.txt              # 📦 Python dependencies
├── run_all.bat                   # ⚡ One-click automation (Windows)
├── data/
│   └── train_data.csv            # 📊 Clean earthquake dataset (37K records)
└── models/
    ├── data_preprocessing_pipeline.pkl  # 💾 Saved preprocessing pipeline
    └── lgb_damage_model.pkl             # 🎯 Trained LightGBM model
```

## Prerequisites

- **Python 3.13+** (or any Python 3.8+)
- **Windows/Linux/macOS** (commands shown for Windows cmd.exe)
- **~2GB free space** for dependencies and data

## Step-by-Step Running Instructions

### Option 1: 🚀 One-Click Run (Recommended for Windows)

1. **Open Command Prompt** in the project directory
2. **Run the automation script:**
   ```cmd
   run_all.bat
   ```
3. **Wait for completion** - the script will:
   - Create preprocessing pipeline
   - Train the model
   - Start the web application
4. **Open your browser** to http://localhost:8501

### Option 2: 📋 Manual Step-by-Step

#### Step 1: Install Dependencies
```cmd
# Navigate to project directory
cd "e:\Internships\Internship_InfosysSpringboard\ImpactSense - Earthquake Impact Prediction\Project"

# Install required Python packages
pip install -r requirements.txt
```

#### Step 2: Train the Model
```cmd
# Run training script (this will also create preprocessing pipeline if needed)
python train.py
```
**Expected Output:**
```
[LightGBM] [Info] Start training from score 5.328670
RMSE: 0.006, R²: 1.000
Saved model and feature list
```

#### Step 3: Start the Web Application
```cmd
# Launch Streamlit app
python -m streamlit run app.py
```
**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

#### Step 4: Use the Application
1. **Open your browser** and navigate to http://localhost:8501
2. **Upload data** in the sidebar - use `data/train_data.csv`
3. **Explore features:**
   - 📊 **Data Tab**: View dataset statistics and visualizations
   - 🔮 **Predict Tab**: Make earthquake impact predictions
   - 🗺️ **Map Tab**: Interactive risk visualization
   - ℹ️ **About Tab**: Learn about the prediction methodology

## Troubleshooting

### Common Issues and Solutions

**1. Python not found error:**
```
'python' is not recognized as an internal or external command
```
**Solution:** Ensure Python is installed and added to PATH, or use full path:
```cmd
"C:\Program Files\Python313\python.exe" train.py
```

**2. Missing packages error:**
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution:** Install dependencies:
```cmd
pip install -r requirements.txt
```

**3. Port already in use:**
```
Port 8501 is already in use
```
**Solution:** Either:
- Stop other Streamlit apps, or
- Use a different port: `streamlit run app.py --server.port 8502`

**4. File not found errors:**
```
FileNotFoundError: models/data_preprocessing_pipeline.pkl
```
**Solution:** Run training first:
```cmd
python train.py
```

### Performance Notes
- **First run**: Takes ~2-3 minutes (model training)
- **Subsequent runs**: Takes ~10 seconds (loads saved model)
- **SHAP explanations**: May take 5-10 seconds to compute
- **Large datasets**: Upload may be slow for files >100MB

## Project Features

### 📊 Data Analysis
- **Dataset Overview**: 37K+ earthquake records
- **Statistical Summary**: Mean, std, min, max for all features
- **Distribution Plots**: Histograms for magnitude, depth, location
- **Correlation Matrix**: Feature relationships visualization

### 🔮 Prediction Engine
- **Input Parameters**: Latitude, Longitude, Depth, Magnitude, etc.
- **ML Model**: LightGBM Regressor (600 estimators, 0.05 learning rate)
- **Output Metrics**: 
  - Damage Potential Score (0-10 scale)
  - Risk Category (Low/Moderate/High)
  - Urban Risk Score (location-adjusted)
- **Explainability**: SHAP feature importance plots

### 🗺️ Interactive Mapping
- **World Map**: Plotly-powered interactive visualization
- **Risk Layers**: Color-coded by damage potential
- **Zoom/Pan**: Explore specific regions
- **Hover Data**: Detailed earthquake information

### 🎯 Risk Assessment Methodology
- **Damage Potential Formula**: `0.6 × magnitude + 0.2 × (700 - depth) / 700 × 10`
- **Risk Categories**:
  - Low: Score < 4.0
  - Moderate: Score 4.0-6.0  
  - High: Score > 6.0
- **Urban Risk**: Adjusted for population exposure based on coordinates

## Technical Details

### Model Performance
- **RMSE**: ~0.006 (very low prediction error)
- **R²**: ~1.000 (excellent fit)
- **Training Time**: ~30 seconds on modern hardware
- **Inference Time**: <1ms per prediction

### Data Pipeline
1. **Feature Selection**: Extract relevant earthquake parameters
2. **Missing Value Imputation**: RandomForest-based for Root Mean Square
3. **Type Filtering**: Focus on earthquake events only
4. **Feature Engineering**: Create damage potential scores
5. **Categorical Encoding**: One-hot encoding for magnitude types and status

### Dependencies
```
pandas           # Data manipulation
numpy            # Numerical computing  
scikit-learn     # ML pipeline and preprocessing
lightgbm         # Gradient boosting model
streamlit        # Web application framework
shap             # Model explainability
matplotlib       # Basic plotting
seaborn          # Statistical visualizations
plotly           # Interactive maps and plots
joblib           # Model serialization
```

## License & Usage

This project is for educational and research purposes. The earthquake data is sourced from public seismic databases. Feel free to use, modify, and extend for your own earthquake risk assessment projects.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify file paths and permissions
4. Check Python version compatibility (3.8+ required)

---

**🌍 Stay Safe and Informed About Earthquake Risks! 🌍**