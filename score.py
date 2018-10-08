import pickle
import pandas as pd
from train_model import DataProcessor
from sklearn.metrics import classification_report


## load pickle file from train_model
pkl = pickle.load(open("pickle_model.pkl", "rb"))

## load locally saved train and test datasets
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv', names =['Survived'])

## use pipeline for prediction on test data
y_predicted = pkl.predict(X_test)
target_names = ['Not-Survived', 'Survived']

## print result
cl_result = classification_report(y_test, y_predicted,target_names=target_names)
print (cl_result)

### save classification result as csv file locally
with open('report.csv', 'w') as f:
    f.write(cl_result)
    f.close()
