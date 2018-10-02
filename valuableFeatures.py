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
# pd.set_option('display.max_columns', None)  
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)

# path = "Datasets/2018Draft.csv"
path = "Datasets/2018DraftExcel.csv"
# path = "Datasets/2018DraftTest.csv"
data = pd.read_csv(path)

pts = data["pts"]
pick = data["pick"]
tier = data["prospect_tier"]


# tier.plot(kind="hist")

# sb.regplot(x="pts", y="prospect_tier", data=data, scatter=True)

# sb.pairplot(data)

# plt.show()

# print (data.head())

# stats_df = pd.DataFrame((data.ix[:,(14,15,20,28)].values), columns=["trb", "ast", "pts", "prospect_tier"] ,dtype=float)
stats_df = pd.DataFrame((data.ix[:,(14,15,20,25,26,27,28)].values), columns=["trb", "ast", "pts","pts_per_g","trb_per_g","ast_per_g","prospect_tier"] ,dtype=float)
stats_df = stats_df.fillna(0.0)
stats_df = stats_df.round(0)
stats_df = stats_df.astype(np.float64)

# data_target = data.ix[:,28].values
# print (stats_df)
# stats_df["group"] = pd.Series(data_target, dtype="category")
# print ("hi")

# sb.pairplot(stats_df, hue='group', palette="hls")
sb.pairplot(stats_df, hue='prospect_tier', palette="husl",  markers=["o", "s", "D"])

# sb.pairplot(stats_df, x_vars=["pts_per_g","trb_per_g","ast_per_g"], y_vars=["prospect_tier"], hue='prospect_tier')

# data.plot(kind="scatter", x="pts", y = "prospect_tier", c=["red"], s=150)

plt.show()




