# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Janagiraman.M
RegisterNumber:  212224230101
*/


import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data
data.info
data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:
DATA:

<img width="917" height="502" alt="1" src="https://github.com/user-attachments/assets/992f8fc8-6489-40ee-8709-7d26bc7b70a1" />

CONFUSION MATRIX:

<img width="125" height="48" alt="2" src="https://github.com/user-attachments/assets/51db8b36-2ddb-4b70-9e95-352befdf4b5c" />

CLASSIFICATION:

<img width="602" height="221" alt="3" src="https://github.com/user-attachments/assets/00d4ef18-8010-4be8-a578-9fbd9a784aeb" />

ACCURACY:

<img width="237" height="38" alt="4" src="https://github.com/user-attachments/assets/880126dd-271d-406e-853c-bb761cea49cd" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
