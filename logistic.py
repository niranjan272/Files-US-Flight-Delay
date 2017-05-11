# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:39:06 2017

@author: Niranjan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May  7 22:48:24 2017

@author: Niranjan
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#Data Load
path = 'F:\\Study\\OneDrive - The University of Texas at Dallas\\02 Study\\12 Startup.ML\\Files\\Input_File\\'
path_file = path+'flight_data.csv'

flight_data = pd.read_csv(path_file)

flight_data['Delayed'] = np.where(flight_data['ARR_DELAY'] >=15, 1,0)
flight_data['Delayed'].value_counts()

print (flight_data['ARR_DELAY'][flight_data['ARR_DELAY'] >= 15].count())
print (flight_data.shape[0] - flight_data.CARRIER_DELAY.isnull().sum().sum())
flight_delay =flight_data['ARR_DELAY'][flight_data['ARR_DELAY'] >= 15].count()/flight_data.shape[0]
print("Percentage of Flights Delayed for More Than 15 Minutes:  {:.2%}".format(flight_delay))


flight_data_filtered = flight_data[['MONTH', 'DAY_OF_MONTH', 'UNIQUE_CARRIER', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'CRS_DEP_TIME', 'DEP_TIME', 'CRS_ARR_TIME','Delayed', 'ARR_TIME', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE']]
flight_data_filtered = flight_data_filtered.dropna()
flight_data_filtered_dummy = pd.get_dummies(flight_data_filtered)
flight_data_shuffle = shuffle(flight_data_filtered_dummy)
flight_data_shuffle.shape

y = flight_data_shuffle['Delayed']
x = flight_data_shuffle.drop('Delayed',1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

X_train.to_csv(path+'X_train.csv')
X_test.to_csv(path+'X_test.csv')
y_train.to_csv(path+'y_train.csv')
y_test.to_csv(path+'y_test.csv')


logistic = LogisticRegression()
count = 0
model_iteraction = 0
while count < len(X_train):
    test_data_x = X_train[count:count+1000]
    test_data_y = y_train[count:count+1000]
    count = count + 1000
    print("Training Model iteration "+str(model_iteraction))
    logistic.fit(test_data_x,test_data_y)
    model_iteraction +=1
    
    
final_logistic = logistic.predict(X_test)
final_logistic = pd.DataFrame(final_logistic)
matrix = confusion_matrix(y_test, final_logistic)
print(matrix)
print("Accuracy Score: ",accuracy_score(y_test,final_logistic))
#pd.crosstab(final_logistic,y_test)


















