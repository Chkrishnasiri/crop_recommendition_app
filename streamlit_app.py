import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load dataset
data = pd.read_csv("real_crop_recommendation_dataset.csv")

# train model
X = data.drop("crop", axis=1)
y = data["crop"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

st.title("🌱 Crop Recommendation System")

st.write("Enter soil and environmental parameters:")

N = st.number_input("Nitrogen (N)", 0, 140, 50)
P = st.number_input("Phosphorus (P)", 0, 100, 40)
K = st.number_input("Potassium (K)", 0, 100, 40)
temperature = st.number_input("Temperature (°C)", 0.0, 40.0, 25.0)
humidity = st.number_input("Humidity (%)", 0, 100, 60)
ph = st.number_input("pH", 0.0, 14.0, 6.5)
rainfall = st.number_input("Rainfall (mm)", 0, 300, 100)

if st.button("Predict Crop"):
    sample = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(sample)[0]
    st.success(f"Recommended Crop: {prediction}")
