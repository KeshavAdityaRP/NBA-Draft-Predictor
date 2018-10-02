import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score, mean_squared_error

def load_data(path, columns):
    length = len(columns)
    size = length - 1 
    prospect_stats = pd.read_csv(path)
    filtered_prospect_stats = prospect_stats[columns]
    filtered_prospect_stats = filtered_prospect_stats.fillna(0.0)
    filtered_prospect_stats = filtered_prospect_stats.round(0)
    filtered_prospect_stats = filtered_prospect_stats.astype(np.float64)
    # X = filtered_prospect_stats.iloc[:, 0:6].values
    # y = filtered_prospect_stats.iloc[:, 6].values
    X = np.array(filtered_prospect_stats.iloc[:, 0:size].values)
    y = np.array(filtered_prospect_stats.iloc[:, size].values)
    return X,y

def cleaning_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    print (X_train.shape)
    print (y_train.shape)
    print (X_train)
    print (y_train)    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def linearRegression(X_train, X_test, y_train, y_test):
    print (X_train.shape)
    print (y_train.shape)
    print (X_train)
    print (y_train)
    model = Sequential()
    model.add(Dense(1, activation='linear', input_dim=21))
    model.compile(loss='mse', optimizer='rmsprop')
    estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=16, verbose=1)
    estimator.fit(X_train, y_train)
    y_test_prediction = estimator.predict(X_test)
    rmse_error = mean_squared_error(y_pred=y_test_prediction, y_true=y_test)
    r2_error = r2_score(y_pred=y_test_prediction, y_true=y_test)
    print ("RMSE Error")
    print (rmse_error) 
    print ("R2 Error")
    print (r2_error)
    # model.fit(X_train, y_train, nb_epoch=100, batch_size=16,verbose=0)
    # model.fit(X_train, y_train, epochs=100, batch_size=16,verbose=1)
    score = model.evaluate(X_test, y_test, batch_size=16)
    print ("Score")
    print (score)




path = "Datasets/recent5DraftClasses.csv"
# columns = ["trb", "ast", "pts","pts_per_g","trb_per_g","ast_per_g","prospect_tier"]
columns = ['fg','fga','fg3','fg3a','ft','fta','orb','trb','ast','stl','blk','tov','pf','pts','fg_pct','fg3_pct','ft_pct','mp_per_g','pts_per_g','trb_per_g','ast_per_g','pick']
X, y = load_data(path, columns)
X_train, X_test, y_train, y_test = cleaning_data(X,y)  
linearRegression(X_train, X_test, y_train, y_test)  