import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy
from sklearn import decomposition
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans


def without_standardisation():
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = Axes3D(fig)

    # X_std = StandardScaler().fit_transform(player_val)

    data = pd.read_csv('Datasets/collegeStats.csv')
    data = pd.read_csv('Datasets/2018Draft.csv')
    players = data[['pts_per_g','trb_per_g','ast_per_g','name']].copy()
    print (players.shape)
    print (players.head())

    # player_val = players.values

    f1 = players['pts_per_g'].values
    f2 = players['trb_per_g'].values
    f3 = players['ast_per_g'].values
    names = players['name'].values
    # X = np.array(list(zip(f1, f2, f3)))
    # plt.scatter(f1, f2, f3, c='black')

    ax.scatter(f1,f2,f3)
    ax.set_xlabel('Point / Game')
    ax.set_ylabel('Rebounds / Game')
    ax.set_zlabel('Assists / Game')

    #give the labels to each point
    for x_label, y_label, z_label, label in zip(f1, f2, f3, names):
        ax.text(x_label, y_label, z_label, label)
    plt.title("Draft Prospects")
    plt.show()

def with_standardisation():
    data = pd.read_csv('Datasets/collegeStats.csv')
    data = pd.read_csv('Datasets/2018Draft.csv')
    playersStats = data[['pts_per_g','trb_per_g','ast_per_g']].copy()
    playersNames = data[['name']].copy()
    playersNames = playersNames.values
    playersStats = playersStats.values
    X_std = StandardScaler().fit_transform(playersStats)
    sklearn_pca = sklearnPCA(n_components=2)
    X_sklearn = sklearn_pca.fit_transform(X_std)
    # print (X_sklearn[0:5,:])
    PCA1 = X_sklearn[:,:1]
    PCA2 = X_sklearn[:,1:]
    # print (PCA1[0:5,:])
    # print (PCA2[0:5,:])

    fig, ax = plt.subplots()
    ax.scatter(PCA1, PCA2)

    for x_label, y_label, label in zip(PCA1, PCA2, playersNames):
        ax.annotate(label, (x_label, y_label))    
    plt.show()

def KMeansClusting():
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')
    # data = pd.read_csv('Datasets/collegeStats.csv')
    # data = pd.read_csv('Datasets/2018Draft.csv')
    data = pd.read_csv('Datasets/last5DraftClasses.csv')    
    playersStats = data[['pts_per_g','trb_per_g','ast_per_g']].copy()
    playersNames = data[['name']].copy()
    playersNames = playersNames.values 
    playersStats = playersStats.values
    X_std = StandardScaler().fit_transform(playersStats)
    sklearn_pca = sklearnPCA(n_components=2)
    X = sklearn_pca.fit_transform(X_std)
    PCA1 = X[:,:1]
    PCA2 = X[:,1:]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    print(centroids)
    print(labels)  

    colors = ["g.","r.","c.","y."]

    fig, ax = plt.subplots()

    for i in range(len(X)):
        # print("coordinate:",X[i], "label:", labels[i])
        plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


    plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

    for x_label, y_label, label in zip(PCA1, PCA2, playersNames):
        ax.annotate(label, (x_label, y_label))  

    plt.show()      


KMeansClusting()
# with_standardisation()