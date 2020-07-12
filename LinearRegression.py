import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

"""
    attributes ["G1", "G2", "G3", "studytime", "failures", "absences"] ~ 78% accuracy

    TODO: Try more or less attributes to increase accuracy
    TODO: Input student number in console then return its predicted and actual grade
"""

#read the csv file, then remove the sep in the csv
data = pd.read_csv("student-mat.csv", sep=';')
print(data.head())


#trim data into attributes that we feel are relevant to the model
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = "G3"

#new data frame where removed G3 column (The attribute)
X = np.array(data.drop([predict], 1))
#Label (What we're trying to predict)
y = np.array(data[predict])

#split each data frame into training and testing sets

"""
we do this because there's no sense in training a model on all data, 
it would simply just memorize the patterns. 

10% of data into test samples, 90% on training data

"""
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)

linear = linear_model.LinearRegression()

#find best fit line foor X and y training data
linear.fit(X_train, y_train)
#get accuracy of model
acc = linear.score(X_test, y_test)

print("Model Accuracy: ",acc)
#coefficient of our 5 variables in y=mx+b
#print("Coefficient: \n", linear.coef_)
#print("Intercept: \n", linear.intercept_)

predictions = linear.predict(X_test)
for x in range(len(predictions)):
    #to visualize predictions and their inputs: print(model's predicted value, inputs, actual grade)
    print("Student",x, ":" ,predictions[x], "-" , y_test[x])









