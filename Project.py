# 🇮🇳 MY INDIA FINAL YEAR PROJECT
# Government Scheme Eligibility Predictor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



np.random.seed(42)

data_size = 500

data = pd.DataFrame({
    "Age": np.random.randint(18, 70, data_size),
    "Annual_Income": np.random.randint(50000, 500000, data_size),
    "Land_Owned_Acres": np.random.randint(0, 10, data_size),
    "Family_Size": np.random.randint(1, 10, data_size),
    "Rural": np.random.choice([0,1], data_size),  # 1 = Rural
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

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)



rf_model = RandomForestClassifier(n_estimators=200)
lr_model = LogisticRegression()

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)



rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
# print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

# print("\nClassification Report (Random Forest):\n")
# print(classification_report(y_test, rf_pred))



# cm = confusion_matrix(y_test, rf_pred)

# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title("Confusion Matrix - Random Forest")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()



feature_importance = pd.Series(rf_model.feature_importances_, index=data.columns[:-1])
feature_importance.sort_values().plot(kind="barh", figsize=(8,6))
plt.title("Feature Importance")
plt.show()



def predict_eligibility(input_data):
    input_scaled = scaler.transform([input_data])
    prediction = rf_model.predict(input_scaled)
    return label_encoders["Eligibility"].inverse_transform(prediction)[0]


example = [35, 120000, 2, 5, 1, 1, 0]  

# Age, Income, Land, Family, Rural, Caste, Employment

# General → 0

# OBC → 1

# SC → 2

print("\nPredicted Eligibility:", predict_eligibility(example))
