import numpy as np
import pandas as pd
from pandas import Series, DataFrame 
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import scipy
from sklearn import decomposition
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
import seaborn as sb

# Quick Look Up
# pick,year,name,university,link,g,mp,fg,fga,fg3,fg3a,ft,fta,orb,trb,ast,stl,blk,tov, pf, pts,fg_pct,fg3_pct,ft_pct,mp_per_g,pts_per_g,trb_per_g,ast_per_g,prospect_tier
# 0,  1,      2,  3,       ,4  , 5, 6, 7, 8, 9,  10,  11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,    22,     23,    24,      25,       26,       27,       ,28

plt.rcParams['figure.figsize'] = (16, 4)
sb.set_style('whitegrid')

def load_data(path, columns):
    prospect_stats = pd.read_csv(path)
    filtered_prospect_stats = prospect_stats[columns]
    filtered_prospect_stats = filtered_prospect_stats.fillna(0.0)
    filtered_prospect_stats = filtered_prospect_stats.round(0)
    filtered_prospect_stats = filtered_prospect_stats.astype(np.float64)
    return filtered_prospect_stats, prospect_stats

def save_data_visulisation(sns_plot):
    sns_plot.savefig("output.png", dpi=1000)

def visulise_data(filtered_prospect_stats,categorise_based_on):
    sns_plot = sb.pairplot(filtered_prospect_stats, hue=categorise_based_on, palette="husl",  markers=["x", "s", "o"])
    # save_data_visulisation(sns_plot)
    plt.show()

def x_vs_y(prospect_stats, x, y, types):
    X = prospect_stats[[x]].copy()
    Y = prospect_stats[[y]].copy()
    playersNames = prospect_stats[['name']].copy()
    X = X.values
    Y = Y.values
    playersNames = playersNames.values 

    fig, ax = plt.subplots()
    if (types == "categoryBased"): 
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan']
        markers = ['o', 'x', 'v' , 's']    
        for i in range(len(X)):
            plt.plot(X[i][0], Y[i][0], colors[Y[i][0]], markersize = 10, marker = markers[Y[i][0]])
        for x_label, y_label, label in zip(X, Y, playersNames):
            ax.annotate(label, (x_label, y_label))       
    else:
        for i in range(len(X)):
            plt.plot(X[i][0], Y[i][0], 'tab:cyan', markersize = 2, marker = 'o')
        for x_label, y_label, label in zip(X, Y, playersNames):
            ax.annotate(label, (x_label, y_label))    
    plt.show()             


# Initialisation

# path = "Datasets/2018DraftExcel.csv"
path = "Datasets/recent5DraftClasses.csv"
columns = ["trb", "ast", "pts","pts_per_g","trb_per_g","ast_per_g","prospect_tier"]
categorise_based_on = 'prospect_tier'
# x = 'pts'
x = 'pts_per_g'
# y = 'prospect_tier'
y = 'pick'

filtered_prospect_stats, prospect_stats = load_data(path, columns)
# visulise_data(filtered_prospect_stats, categorise_based_on)
x_vs_y(prospect_stats,x, y, "scatterBased")

