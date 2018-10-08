
import pandas as pd

url_train="https://raw.githubusercontent.com/gpsingh12/kaggle-titanic/master/train.csv"
url_test="https://raw.githubusercontent.com/gpsingh12/kaggle-titanic/master/test.csv"

def load_data(url):
    '''Functionn downloads data from github using url'''
    df=pd.read_csv(url)
    return (df)

train = load_data(url_train) # loads training data and save to dataframe train
print ('Training dataframe is loaded with dimensions: {0}'.format(train.shape))
print ('----------------------------------------------------------------------')
test  = load_data(url_test)  # loads test data and save to dataframe test
print ('Test dataframe is loaded with dimensions: {0}'.format(test.shape))
