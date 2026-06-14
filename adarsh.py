import pandas as pd
data=pd.read_csv("student-lifestyle-and-stress-dataset.csv")
print(data.head())
print(f"Number of duplicate rows: {data.duplicated().sum()}")
data=data.drop_duplicates()
print(f"Number of duplicate rows after removal: {data.duplicated().sum()}")
print(data.isnull().mean()*100)
data['Sleep_Hours'] = data['Sleep_Hours'].fillna(data['Sleep_Hours'].median())
data['Study_Hours'] = data['Study_Hours'].fillna(data['Study_Hours'].median())
data['Social_Media_Hours'] = data['Social_Media_Hours'].fillna(data['Social_Media_Hours'].median())
data['Attendance'] = data['Attendance'].fillna(data['Attendance'].median())
data['Exam_Pressure'] = data['Exam_Pressure'].fillna(data['Exam_Pressure'].median())
data['Month'] = data['Month'].fillna(data['Month'].median())
data['Family_Support'] = data['Family_Support'].fillna(data['Family_Support'].median())
data['Student_Type'] = data['Student_Type'].fillna(data['Student_Type'].mode()[0])
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder             
label_encoder = LabelEncoder()
data['Student_Type']=label_encoder.fit_transform(data['Student_Type'])
x=data.drop(columns=["Stress_Level"])
y=(data[['Stress_Level']])

from sklearn.preprocessing import StandardScaler           
x_scaler=StandardScaler()
x_scaled=x_scaler.fit_transform(x)

from sklearn.model_selection import train_test_split     
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression         
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print('coefficient :',model.coef_)
print('intercept:',model.intercept_)

from sklearn.metrics import accuracy_score             
accuracy=accuracy_score(y_test,y_predict)
print('accuracy :',accuracy)

import joblib                                             
joblib.dump(model, "model.pkl") 
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(label_encoder, "label_encoder.pkl") 
