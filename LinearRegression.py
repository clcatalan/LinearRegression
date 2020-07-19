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
#print(data.head())


def read_input():
    idx = int(input('Enter Student Number: '))
    if idx >= 0 and idx < len(data):
        return idx
    else:
        print('Invalid Input')
        return -1


#def clean_data(data):
   
stdNo = read_input()
#clean_data(data)
avg = [0] * len(data)
for x, row in data.iterrows():
    avg[x] = (row["G1"] + row["G2"] + row["G3"]) / 3


data["avg"] = avg

#change text values into numbers
data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})
data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
data['address'] = data['address'].map({'U': 1, 'R': 0})
data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})

#this is to see the attributes are present in the top performing students. If certain values of attributes appear often in them, then it probably means those attributes are relevant
print(data.nlargest(10, 'avg'))

#address, pstatus, Medu, Dalc, Walc

#trim data into attributes that we feel are relevant to the model
#attributes = ["G1", "G2", "G3", "studytime", "failures", "absences"] #80% accurate
attributes = ["G1", "G2", "G3","studytime", "failures", "absences", "schoolsup", "famsup","address", "Pstatus"]
data = data[attributes]    

predict = "G3"
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
leftAlign = 'Student No'
center = 'Predicted Grade'
rightAlign = 'Actual Grade'
print(f"{leftAlign:<15}{center:^10}{rightAlign:>15}")
print(f"{stdNo:<15}{predictions[stdNo]:^10}{y_test[stdNo]:>15}")


#for x in range(len(predictions)):
    #to visualize predictions and their inputs: print(model's predicted value, inputs, actual grade)
    #print("Student",x, ":" ,predictions[x], "-" , y_test[x])









