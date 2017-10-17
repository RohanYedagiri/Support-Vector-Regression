# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('Position_Salaries.csv')

# separate x,y
X = dataset.drop(['Position','Salary'],axis=1)
y = dataset.Salary

# Feature scaling
from sklearn.preprocessing import StandardScaler
sx = StandardScaler()
sy = StandardScaler()

X = sx.fit_transform(X)
y = sy.fit_transform(y)

# fitting svr
from sklearn.svm import SVR
regsvr = SVR(kernel='rbf')
regsvr.fit(X,y)

# predict
y_pred = regsvr.predict(sx.transform(np.array([[6.5]])))
y_pred = sy.inverse_transform(y_pred)

#viz
plt.scatter(X,y,color='red')
plt.plot(X, regsvr.predict(X),color='blue')
plt.title('linear regression 1 predictions')
plt.xlabel('Position level')
plt.ylabel('salaries')
plt.show()