import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
print("my name")

data=pd.read_csv("taxi.csv")
#print(data.head())

data_x=data.iloc[:,:-1].values
data_y=data.iloc[:,-1].values
# print(data_y)
X_train,X_test,y_train,y_test=train_test_split(data_x,data_y,test_size=.30,random_state=0)

reg=LinearRegression()
reg.fit(X_train,y_train)
print("Train score",reg.score(X_train,y_train))
print("Test score",reg.score(X_test,y_test))


pickle.dump(reg,open('taxi.pkl','wb'))




model=pickle.load(open('taxi.pkl','rb'))

print(model.predict([[80,1770000,6000,85]]))
