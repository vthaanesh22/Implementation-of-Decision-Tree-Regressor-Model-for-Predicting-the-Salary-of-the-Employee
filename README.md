# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. Calculate Mean square error,data prediction and r2.
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Thaanesh.V
RegisterNumber:  212223230228
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
# Data Head:
![image](https://github.com/user-attachments/assets/0a5ff489-b161-4da0-a3ce-da9d7ff634f1)
# Data Info
![image](https://github.com/user-attachments/assets/b3c44418-c20e-4689-92cc-09808dabec2d)
# isnull().sum()
![image](https://github.com/user-attachments/assets/05e76561-fe93-497b-bf5d-136786d1394c)
# Data Head for salary:
![image](https://github.com/user-attachments/assets/36e500f1-a75e-42c0-993a-0855f074d199)
# Mean Squared Error :
![image](https://github.com/user-attachments/assets/f1bc9571-b5d1-414c-95b8-f0bcb9876ac8)
# r2 Value:
![image](https://github.com/user-attachments/assets/89285657-e49f-4813-b985-e878c7ae2671)
# Data prediction :
![image](https://github.com/user-attachments/assets/875ebf4e-ef9b-475c-accb-3ebd68d959d2)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
