#Time series forecasting - linear regression 

import math
import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


#To print all the rows
pd.set_option('display.max_rows', None)
data = pd.read_csv('infy.csv')
#print(data.head())


# # 'Shift function' is used to shift rows above as required .here we have to shift by -3 above. Shift rows above use (-) and shift rows below(+)
data['Label'] = data['Close Price'].shift(-3)
#print(data[['Close Price','Label']].head())


total = data[['Open Price','High Price','Low Price','Close Price','Label']]
#print(total.head())


x = np.array(data.drop(['Close Price','Label'], axis=1))
# #X = FEATURES AND Y = LABEL
X = np.array(data[['Open Price','High Price','Low Price']])
X_lately = X[-3:]
#print(X_lately)
X = X[:-3]
#print(X)

y = np.array(data['Label'])
y = y[:-3]
#print(y)



# # (.shape) gives dimensions 
# #print(X.shape, y.shape)


# # train_test_split is used to divide the data into training and testing set by user defined ratio .it takes 3 parameters(x,y,ratio)
# # training set - higher part   and    testing set - lower part
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# #print(X_train, X_test, y_train, y_test)
# #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # (clf) is the object of class linear regression
clf = LinearRegression()


# # fit() func is a func inside linear regression class that will train the model
clf.fit(X_train, y_train)


# # Score() func gives coefficient of determination
confidence = clf.score(X_test, y_test)          
#print('confidence:', confidence)


# # Predict() func takes the input and returns the output
forecast_set = clf.predict(X_lately)
print(forecast_set)
