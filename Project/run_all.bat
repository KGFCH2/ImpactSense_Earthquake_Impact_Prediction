@echo off
echo ========================================
echo ImpactSense - Earthquake Impact Prediction
echo Running complete pipeline...
echo ========================================

echo.
echo [1/4] Creating preprocessing pipeline...
"C:\Program Files\Python313\python.exe" safe_preprocess.py
if errorlevel 1 (
    echo ✗ Preprocessing failed - aborting.
    pause
    exit /b 1
)

echo.
echo [2/4] Training LightGBM model...
"C:\Program Files\Python313\python.exe" train.py
if errorlevel 1 (
    echo ✗ Training failed - aborting.
    pause
    exit /b 1
)

echo.
echo [3/4] Verifying output files...
"C:\Program Files\Python313\python.exe" -c "import os; print('Pipeline:', '✓' if os.path.exists('models\\\\data_preprocessing_pipeline.pkl') else '✗'); print('Model:', '✓' if os.path.exists('models\\\\lgb_damage_model.pkl') else '✗')"

echo.
echo [4/4] Starting Streamlit app...
echo Opening http://localhost:8501 in your browser...
echo Upload data/train_data_clean.csv in the sidebar to test the app.
echo Press Ctrl+C to stop the server.
echo.
"C:\Program Files\Python313\python.exe" -m streamlit run app.py