# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas 
 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Archana.k
RegisterNumber:  212222240011
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## df.head()

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/b9f7202d-5161-41bf-8ea4-c173ebde3bb5)

## df.tail()

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/538a10c9-ecd8-47ef-8e7e-5a8e7a84e21b)

## Array value of x

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/75a378a5-2c7f-4c06-8270-ba904e3429c6)

## Array value of y

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/0f9a9ceb-70d8-4d96-a1e5-08172e96db95)

## Values of Y prediction

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/b1a00f55-9ada-4633-aba6-bbacb5bff6f9)

## Array values of Y test
 
![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/1ac194e7-784c-4c45-9603-62c3cd3da382)

## Training Set Graph
 
![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/423db861-1326-494a-a067-d9c63534bbe0)

## Test Set Graph

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/8cfeab38-4b65-4422-87f7-d18663f51009)

## Values of MSE, MAE and RMSE

![image](https://github.com/22009150/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118708624/a9866fa6-523a-485e-a35a-5bc147d1a876)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
