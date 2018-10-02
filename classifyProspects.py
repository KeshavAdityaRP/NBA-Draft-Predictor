import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


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

def NN_model(X_train, X_test, y_train, y_test):
    print (X_train.shape)
    print (y_train.shape)
    print (X_train)
    print (y_train)
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(20,activation = 'relu', input_dim=21))
    # Adding the second hidden layer
    classifier.add(Dense(20, activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(20, activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(20, activation = 'relu'))            
    # Adding the output layer    
    classifier.add(Dense(3, activation ='softmax'))  
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) 
    # Fitting our model 
    classifier.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose = 1, shuffle=True)
    print ("Hi")   
    # serialize weights to HDF5
    classifier.save_weights("labelProspectModel.h5")
    print("Saved model to disk")
    loss, acc = classifier.evaluate(X_test, y_test, batch_size=32)
    print('Test Loss:', loss)
    print('Test Accuracy:', acc) 
    # Predicting the Test set results
    # y_pred = classifier.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)  

path = "Datasets/recent5DraftClasses.csv"
# columns = ["trb", "ast", "pts","pts_per_g","trb_per_g","ast_per_g","prospect_tier"]
columns = ['fg','fga','fg3','fg3a','ft','fta','orb','trb','ast','stl','blk','tov','pf','pts','fg_pct','fg3_pct','ft_pct','mp_per_g','pts_per_g','trb_per_g','ast_per_g','prospect_tier']
X, y = load_data(path, columns)
X_train, X_test, y_train, y_test = cleaning_data(X,y)  
NN_model(X_train, X_test, y_train, y_test)  