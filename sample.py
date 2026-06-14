import joblib
model = joblib.load("model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

student_type = input("Enter Student Type (school/college/working_student): ")
sleep_hours = float(input("Enter Sleep Hours: "))
study_hours = float(input("Enter Study Hours: "))
social_media_hours = float(input("Enter Social Media Hours: "))
attendance = float(input("Enter Attendance: "))
exam_pressure = float(input("Enter Exam Pressure: "))
month = float(input("Enter Month (1-12): "))
family_support = float(input("Enter Family Support: "))
student_type_encoded = encoder.transform([student_type])[0]

import pandas as pd
new_data = pd.DataFrame([[student_type_encoded,sleep_hours,study_hours,social_media_hours,attendance,exam_pressure,month,family_support]], columns=['Student_Type','Sleep_Hours','Study_Hours','Social_Media_Hours','Attendance','Exam_Pressure','Month','Family_Support'])
print("\nuser given Data:",new_data)
new_data_scaled = x_scaler.transform(new_data)
predicted_stress = model.predict(new_data_scaled)

print("\nPredicted Stress Level:", predicted_stress[0])




