import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

# stimu function
def atan(x): 
    return tf.atan(x)

# basic config

class conf:
    instrument = '600000' #code of shares
    start_date = '2005-01-01'
    split_date = '2015-01-01' #before split: training data
    end_date = '2015-12-01' #after split: test data
    fields = ['close', 'open', 'high', 'low', 'volume', 'amount'] #amount = close*volume
    seq_len = 30 #each sample
    batch = 100 #one gradient descent with 100 samples 

def getData():
    # geting data from tushare and preprocessing

    df = ts.get_k_data(conf.instrument, conf.start_date, conf.end_date)
    #df.to_csv("600000data.csv", index=False, sep=',')
    df['amount'] = df['close']*df['volume']
    df['return'] = df['close'].shift(-5) / df['close'].shift(-1) - 1 #return(yield rate) = close price 5 days after / close price tomorrow
    #df['return'] = df['return'].apply(lambda x:np.where(x>=0.2,0.2,np.where(x>-0.2,x,-0.2)))
    df['return'] = df['return']
    df.dropna(inplace=True)
    dftime = df['date'][df.date>=conf.split_date]
    df.reset_index(drop=True, inplace=True)
    scaledf = df[conf.fields]
    traindf = df[df.date<conf.split_date]

    train_input = []
    train_output = []
    test_input = []
    test_output = []
    for i in range(conf.seq_len-1, len(traindf)):
        a = scale(scaledf[i+1-conf.seq_len:i+1])
        train_input.append(a)
        c = df['return'][i]
        train_output.append(c)
    
    for i in range(len(traindf), len(df)):
        a = scale(scaledf[i+1-conf.seq_len:i+1])
        test_input.append(a)
        c = df['return'][i]
        test_output.append(c)
    
    train_m = len(train_output)
    test_m = len(test_output)
    
    train_input = np.array(train_input).reshape(train_m,-1)
    test_input = np.array(test_input).reshape(test_m,-1)
    
    train_output = np.array(train_output)
    test_output = np.array(test_output)
    
    return train_input,train_output,test_input,test_output

    
def draw(y_test,y_pred, model_name):

    plt.title('Return Rate prediction---' + model_name)

    xindex = range(len(y_test))
    plt.plot(xindex, y_test, color = 'red', label='ground truth')
    plt.plot(xindex, y_pred, color = 'blue', label='prediction')
    plt.xlabel('Date')
    plt.ylabel('Return Rate')
    plt.legend(('ground truth','prediction'))
    plt.show()
    
def doSVR(x_train, y_train, x_test, y_test):


    model = SVR()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    print('EVS of SVR:',explained_variance_score(y_test,y_pred))
    print('MAE of SVR:',median_absolute_error(y_test,y_pred))
    print('MSE of SVR:',mean_squared_error(y_test,y_pred))
    print('R2 of SVR:',r2_score(y_test,y_pred))
    
    draw(y_test,y_pred,'SVR')

def doDecision_Tree(x_train, y_train, x_test, y_test):


    model = DecisionTreeRegressor()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    print('EVS of DecisionTreeRegressor:',explained_variance_score(y_test,y_pred))
    print('MAE of DecisionTreeRegressor:',median_absolute_error(y_test,y_pred))
    print('MSE of DecisionTreeRegressor:',mean_squared_error(y_test,y_pred))
    print('R2 of DecisionTreeRegressor:',r2_score(y_test,y_pred))
    
    draw(y_test,y_pred,'DecisionTreeRegressor')
    
def doLR(x_train, y_train, x_test, y_test):


    model = LinearRegression()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    print('EVS of LinearRegression:',explained_variance_score(y_test,y_pred))
    print('MAE of LinearRegression:',median_absolute_error(y_test,y_pred))
    print('MSE of LinearRegression:',mean_squared_error(y_test,y_pred))
    print('R2 of LinearRegression:',r2_score(y_test,y_pred))
    
    draw(y_test,y_pred,'LinearRegression')

def doKNN(x_train, y_train, x_test, y_test):


    model = KNeighborsRegressor()
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print('EVS of KNeighborsRegressor:',explained_variance_score(y_test,y_pred))
    print('MAE of KNeighborsRegressor:',median_absolute_error(y_test,y_pred))
    print('MSE of KNeighborsRegressor:',mean_squared_error(y_test,y_pred))
    print('R2 of KNeighborsRegressor:',r2_score(y_test,y_pred))
    
    draw(y_test,y_pred,'KNeighborsRegressor')

def doRF(x_train, y_train, x_test, y_test):


    model = RandomForestRegressor(n_estimators=30)
    
    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print('EVS of RandomForestRegressor:',explained_variance_score(y_test,y_pred))
    print('MAE of RandomForestRegressor:',median_absolute_error(y_test,y_pred))
    print('MSE of RandomForestRegressor:',mean_squared_error(y_test,y_pred))
    print('R2 of RandomForestRegressor:',r2_score(y_test,y_pred))
    
    draw(y_test,y_pred,'RandomForestRegressor')

def doAdaBoostRegressor(x_train, y_train, x_test, y_test):


    model = AdaBoostRegressor(n_estimators=30)

    model.fit(x_train,y_train)
    
    y_pred = model.predict(x_test)
    
    print('EVS of AdaBoostRegressor:',explained_variance_score(y_test,y_pred))
    print('MAE of AdaBoostRegressor:',median_absolute_error(y_test,y_pred))
    print('MSE of AdaBoostRegressor:',mean_squared_error(y_test,y_pred))
    print('R2 of AdaBoostRegressor:',r2_score(y_test,y_pred))
    draw(y_test,y_pred,'AdaBoostRegressor')

if __name__ == '__main__':


    train_input,train_output,test_input,test_output = getData()


    
    doSVR(train_input,train_output,test_input,test_output)
    doDecision_Tree(train_input,train_output,test_input,test_output)
    doLR(train_input,train_output,test_input,test_output)
    doKNN(train_input,train_output,test_input,test_output)
    doRF(train_input,train_output,test_input,test_output)
    doAdaBoostRegressor(train_input,train_output,test_input,test_output)
    

    


