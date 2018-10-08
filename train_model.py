
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


def std_dat(file):
    ''' Function loads the csv file check unique values for categorical variables
        in an effort to standardize the categories if there are unstandardized values. For example: Categories in column Sex
        male, Male, M corresponds to same value'''
    df = pd.read_csv(file)
    print('Unique values in column Sex:{0}'.format(df.Sex.unique()))
    print('Unique values in column Survived: {0}'.format(df.Survived.unique()))
    print('Unique values in column Pclass :{0}'.format(df.Pclass.unique()))
    print('Unique values in column SibSp:{0}'.format(df.SibSp.unique()))
    print('Unique values in column Parch:{0}'.format(df.Parch.unique()))
    print('Unique values in column Embarked:{0}'.format(df.Embarked.unique()))

    return



def proc_dat(file):
    ''' Function reads locally saved csv file and process data for
        classifier. Drop unnecessary columns, Convert categorical to binary,
        and impute missing values with median'''
    df = pd.read_csv(file)  # read csv
    
    df = df.drop(['PassengerId','Name','Cabin','Ticket'], 1) # drop columns
    df['Sex'] = pd.get_dummies(df['Sex'])   # convert into binary, create dummy
    df['Embarked'] = pd.get_dummies(df['Embarked'])     # create dummy variable
    df = df.fillna(df.median())         # impute missing values with median

    return (df)

def dat_chk(file):
    '''Function perform a data check, if dataframe is loaded correctly,
       print dimensions, column names and missing values (if exists) '''
    df = proc_dat(file)
    print ('Training dataframe is loaded with dimensions: {0}'.format(df.shape))
    print ('Sum of missing values in columns:\n{0}'.format(df.isnull().sum(axis=0)))
    return (df)


## Load the training dataset and pass through above functions 
#train = dat_chk('C:\\Users\\Gurpreet\\Documents\\DATA622\\hw2\\train.csv')


#  class to transform custom function inside pipeline
#Ref: https://stackoverflow.com/questions/25250654/
# how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline/25254187

class DataProcessor(object):

    def transform(self, df1):
        #cols = X[:,2:4] # column 3 and 4 are "extracted"
        df1 = df1.drop(['PassengerId','Name','Cabin','Ticket'], 1)
        df1['Sex'] = pd.get_dummies(df1['Sex'])
        df1['Embarked'] = pd.get_dummies(df1['Embarked'])
        df1 = df1.fillna(df1.median())
        return df1

    def fit(self, df, y=None):
        return self


## initiate steps for pipeline using the class and classifier for logistic regression
steps = [('Data Processing',DataProcessor()),('Logistic Regression', LogisticRegression())]
pipeline = Pipeline(steps)


train=pd.read_csv('https://raw.githubusercontent.com/gpsingh12/kaggle-titanic/master/train.csv')
X = train.drop('Survived', axis=1)
y = train['Survived']

## split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

### save test data localy to check accuracy of classifier
X_test.to_csv('X_test.csv',  index = False)

y_test.to_csv('y_test.csv', index = False)

## create model using pipeline
model = pipeline.fit(X_train, y_train)

## score 
score = model.score(X_test,y_test)


## save as a pickle file locally
pkl_file = "pickle_model.pkl"  
file = open(pkl_file, 'wb')
pickle.dump(model, file)
file.close()




