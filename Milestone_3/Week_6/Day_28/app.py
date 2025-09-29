import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details to predict survival chances.")

# Sidebar for navigation
mode = st.sidebar.radio("Choose Mode:", ["Prediction", "Data Exploration"])

if mode == "Prediction":
    st.header("🧑 Passenger Information")

    pclass = st.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    sex = st.radio("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 25)
    sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
    fare = st.slider("Fare Paid", 0, 500, 50)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    # Convert inputs
    sex = 0 if sex == "male" else 1
    embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("✅ Prediction Result:")
    if prediction == 1:
        st.success(f"Passenger is likely to **Survive** (Probability: {probability:.2f})")
    else:
        st.error(f"Passenger is likely **Not Survive** (Probability: {1-probability:.2f})")

elif mode == "Data Exploration":
    st.header("📊 Titanic Data Exploration")

    df = pd.read_csv("titanic.csv")
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    df.dropna(inplace=True)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Histogram
    st.write("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # Scatter plot
    st.write("### Fare vs Age by Survival")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df, ax=ax)
    st.pyplot(fig)
