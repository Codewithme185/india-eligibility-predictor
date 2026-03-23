import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# TRAIN MODEL (same as your code)
# -----------------------------
np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "Age": np.random.randint(18, 70, data_size),
    "Annual_Income": np.random.randint(50000, 500000, data_size),
    "Land_Owned_Acres": np.random.randint(0, 10, data_size),
    "Family_Size": np.random.randint(1, 10, data_size),
    "Rural": np.random.choice([0,1], data_size),
    "Caste_Category": np.random.choice(["General","OBC","SC","ST"], data_size),
    "Employment_Status": np.random.choice(["Unemployed","Self-Employed","Private","Government"], data_size)
})

def eligibility(row):
    if row["Annual_Income"] < 150000 and row["Rural"] == 1:
        return "Eligible"
    elif row["Annual_Income"] < 250000:
        return "Partially Eligible"
    else:
        return "Not Eligible"

data["Eligibility"] = data.apply(eligibility, axis=1)

label_encoders = {}

for col in ["Caste_Category", "Employment_Status", "Eligibility"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("Eligibility", axis=1)
y = data["Eligibility"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_scaled, y)

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("MACHINE LEARNING MODEL FOR PUBLIC BENEFIT ELIGIBILITY DETECTION")

st.write("Fill the details below:")

age = st.slider("Age", 18, 70, 25)
income = st.number_input("Annual Income", 50000, 500000, 100000)
land = st.slider("Land Owned (Acres)", 0, 10, 1)
family = st.slider("Family Size", 1, 10, 4)

rural = st.selectbox("Area", ["Urban", "Rural"])
caste = st.selectbox("Caste Category", ["General", "OBC", "SC", "ST"])
employment = st.selectbox("Employment Status", ["Unemployed","Self-Employed","Private","Government"])

# Convert inputs
rural_val = 1 if rural == "Rural" else 0
caste_val = label_encoders["Caste_Category"].transform([caste])[0]
emp_val = label_encoders["Employment_Status"].transform([employment])[0]

input_data = [[age, income, land, family, rural_val, caste_val, emp_val]]

# Prediction
if st.button("Check Eligibility"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    result = label_encoders["Eligibility"].inverse_transform(prediction)[0]

    st.success(f"Result: {result}")
