import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle


print("-------Student Final Grade Predicter-------")
#read the csv file, then remove the sep in the csv
data = pd.read_csv("student-mat.csv", sep=';')
#print(data.head())


#------------function definitions--------
"""def readInput():
    idx = int(input('Enter Student Number: '))
    print(len(data))
    if idx >= 0 and idx < len(data):
        return idx
    else:
        print('Invalid Input')
        return -1"""

def exportBestAndWorstToCsv():
    compression_opts = dict(method='zip',
                        archive_name='top-performers.csv')  
    data.nlargest(20, 'avg').to_csv('top.zip', index=False,
            compression=compression_opts)  
    compression_opts = dict(method='zip',
                            archive_name='bottom10.csv')  
    data.nsmallest(20, 'avg').to_csv('worst-performers.zip', index=False,
            compression=compression_opts)  

def cleanData():
    #change text values into numbers
    data['address'] = data['address'].map({'U': 1, 'R': 0})
    data['Pstatus'] = data['Pstatus'].map({'T': 1, 'A': 0})
    data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})
    data['internet'] = data['internet'].map({'yes': 1, 'no': 0})


def printPredictions():
    leftAlign = 'Student No'
    center = 'Predicted Grade'
    rightAlign = 'Actual Grade'


    print(len(predictions))
    print(len(y_test))
    print("-----------------------RESULTS-----------------------")
    print("| Model Accuracy: ",acc)
    print("|-------")
    print(f"| {leftAlign:<15}{center:^10}{rightAlign:>15}")
    for i in range(len(predictions)):
        print(f"| {i:<15}{predictions[i]:^10}{y_test[i]:>15}")
    print("-----------------------------------------------------")

#----------start of app----------

#stdNo = readInput()
avg = [0] * len(data)
for x, row in data.iterrows():
    avg[x] = (row["G1"] + row["G2"] + row["G3"]) / 3


data["avg"] = avg
exportBestAndWorstToCsv()
cleanData()


#trim data into attributes that we feel are relevant to the model
attributes = ["G1", "G2", "G3", "studytime", "failures", "absences", "romantic", "traveltime", "internet"]
data = data[attributes]    
predict = "G3"
#Label (What we're trying to predict)
X = np.array(data.drop(columns=[predict]))
y = np.array(data[predict])

#split each data frame into training and testing sets

"""
we do this because there's no sense in training a model on all data, 
it would simply just memorize the patterns. 

10% of data into test samples, 90% on training data

"""
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size = 0.1)

linear = linear_model.LinearRegression()

#find best fit line for X and y training data
linear.fit(X_train, y_train)
#get accuracy of model
acc = linear.score(X_test, y_test)

predictions = linear.predict(X_test)
printPredictions()












