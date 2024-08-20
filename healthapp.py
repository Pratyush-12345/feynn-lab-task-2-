import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('KaggleV2-May-2016.csv')

# Convert categorical features to numeric
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# Feature engineering
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['WaitingTime'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

# Select features and target
X = df[['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'WaitingTime']]
y = df['No-show']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

import streamlit as st

# Title of the app
st.title('Predictive Healthcare Analytics: No-show Prediction')

# Sidebar for user input
st.sidebar.header('Patient Information')
gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
age = st.sidebar.slider('Age', 0, 100, 30)
scholarship = st.sidebar.selectbox('Scholarship', ('No', 'Yes'))
hypertension = st.sidebar.selectbox('Hypertension', ('No', 'Yes'))
diabetes = st.sidebar.selectbox('Diabetes', ('No', 'Yes'))
alcoholism = st.sidebar.selectbox('Alcoholism', ('No', 'Yes'))
handicap = st.sidebar.selectbox('Handicap', ('No', 'Yes'))
sms_received = st.sidebar.selectbox('SMS Received', ('No', 'Yes'))
waiting_time = st.sidebar.slider('Waiting Time (Days)', 0, 30, 7)

# Convert categorical inputs to numerical format
gender = 1 if gender == 'Male' else 0
scholarship = 1 if scholarship == 'Yes' else 0
hypertension = 1 if hypertension == 'Yes' else 0
diabetes = 1 if diabetes == 'Yes' else 0
alcoholism = 1 if alcoholism == 'Yes' else 0
handicap = 1 if handicap == 'Yes' else 0
sms_received = 1 if sms_received == 'Yes' else 0

# Predict using the trained model
input_data = [[gender, age, scholarship, hypertension, diabetes, alcoholism, handicap, sms_received, waiting_time]]
prediction = model.predict(input_data)

# Output prediction
if prediction == 0:
    st.success("The patient is likely to show up for the appointment.")
else:
    st.error("The patient is likely to miss the appointment.")

