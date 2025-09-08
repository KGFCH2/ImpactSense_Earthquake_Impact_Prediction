# 🌍 ImpactSense: Earthquake Impact Prediction  

## 📌 Project Statement  
ImpactSense is a **machine learning-based predictive system** designed to estimate the impact of earthquakes in terms of **magnitude, damage level, or risk zone classification**.  

Using **geophysical and environmental data** (latitude, longitude, depth, seismic wave features, and geological parameters), the model helps in **disaster preparedness, urban planning, and emergency response**.  

---

## 🚀 Use Cases  

### 🏙 Urban Risk Assessment  
- **Description:** Predict earthquake impact in populated regions based on historical and geophysical data.  
- **Example:** Identify which areas are at higher risk during a 5.5 magnitude earthquake.  

### 🏗 Infrastructure Planning  
- **Description:** Guide construction policies in high-risk seismic zones.  
- **Example:** Predict risk based on soil density and proximity to fault lines.  

### 🚑 Government Disaster Response  
- **Description:** Prioritize rescue and aid delivery based on predicted severity.  
- **Example:** Rank regions for emergency support immediately after an earthquake.  

---

## 🎯 Expected Outcomes  
By the end of this project, you will:  
- Understand **seismic data** and its role in earthquake impact prediction.  
- Perform **data preprocessing** and **feature engineering**.  
- Train and evaluate **classification & regression models**.  
- (Optional) Build a **user-friendly prediction interface**.  
- Document results with **charts and reports**.  

---

## 📂 Dataset  
- **Source:** [Kaggle](https://www.kaggle.com)  

---

## 🏗 System Architecture  
📌 Refer to `system_architecture.png` in the project files.  

---

## 🔧 Modules  

### 1️⃣ Data Exploration & Cleaning  
- Load dataset, handle missing values, remove duplicates  
- Visualize features: depth, magnitude, latitude, longitude  

### 2️⃣ Feature Engineering  
- Scale/normalize numeric features  
- Create geospatial clusters & risk scores  
- Encode categorical variables  

### 3️⃣ Model Development  
- Train baseline: Logistic Regression, Decision Tree  
- Advanced models: Random Forest, XGBoost, Gradient Boosting  

### 4️⃣ Model Evaluation  
- Metrics: Accuracy, Precision, Recall, F1, MAE/MSE  
- Confusion matrix & feature importance  

### 5️⃣ User Interface (Optional)  
- Built with **Streamlit** or **FastAPI**  
- Input: Magnitude, Depth, Region, Soil Type  
- Output: Impact Prediction / Risk Category  

---

## 📅 Milestones  

| **Milestone** | **Week** | **Tasks** |
|---------------|----------|------------|
| **Milestone 1** | Week 1 | Project setup, dataset exploration, feature distribution, mapping locations |
|               | Week 2 | Data preprocessing, handle missing values, feature engineering |
| **Milestone 2** | Week 3 | Train baseline models: Logistic Regression, Decision Tree (basic accuracy/MAE) |
|               | Week 4 | Train advanced models: Random Forest, Gradient Boosting (cross-validation & hyperparameter tuning) |
| **Milestone 3** | Week 5 | Model evaluation & explainability: confusion matrix, error plots, feature importance, SHAP values |
|               | Week 6 | Build prototype UI (input → impact prediction) |
| **Milestone 4** | Week 7 | Testing & improvements (edge cases, refine model & UI) |
|               | Week 8 | Final report & presentation (charts, visuals, results, slides, PDF) |

---

## 📝 Evaluation Criteria  

✅ Completion of Milestones  
✅ Prediction Accuracy & Robustness  
✅ Clear Documentation & Visuals  
✅ Presentation & Explanation  

---

## 📊 Model Performance Metrics  

| **Category** | **Metrics** |
|--------------|-------------|
| **Classification** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Regression** | Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score |
| **Explainability** | Feature Importance (Depth, Magnitude, Soil Type), SHAP Value Plots, Training vs Validation Curves |

---

## ⚙️ Tech Stack  
- **Language:** Python 🐍  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost  
- **UI (Optional):** Streamlit / FastAPI  
- **Visualization:** Matplotlib, Seaborn, Plotly  

---

## 📌 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/KGFCH2/ImpactSense_Earthquake_Impact_Prediction.git
   cd ImpactSense_Earthquake_Impact_Prediction
