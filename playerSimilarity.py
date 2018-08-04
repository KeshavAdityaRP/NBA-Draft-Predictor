import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import csv 

def load_data(filename):
    print ('Loading data...')
    df = pd.read_csv(filename)
    print ('Data has been loaded sucessfully')
    player_name = df[['Name']].copy()
    name = player_name.values
    player_stats = df[['Games Played','MIN','PTS','FGM','FGA','FG%','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','REB','AST','STL','BLK','TOV','PF','EFF','AST/TOV','STL/TOV']].copy()
    # player_stats = df[['PTS','FGM']].copy()
    player_stats = player_stats.astype(np.float64)
    stats = player_stats.values
    stats = stats / stats.max(axis=0)
    return name, stats

def load_nba_data(filename):
    print ('Loading data...')
    df = pd.read_csv(filename,error_bad_lines=False) # ,error_bad_lines=False
    print ('Data has been loaded sucessfully')
    player_name = df[['name']].copy()
    name = player_name.values
    # player_stats = df[['G','MP','FG','FGA','FGP','2P','2PA','2PP','3P','3PA','3PP','FT','FTA','FTP','TRB','AST','STL','BLK','TOV','PF','PTS']].copy()
    player_stats = df[['g','mp','fg','fga','fg3','fg3a','ft','fta','orb','trb','ast','stl','blk','tov','pf','pts','fg_pct','fg3_pct','ft_pct','mp_per_g','pts_per_g','trb_per_g','ast_per_g']].copy()
    # print (player_stats.iloc[23])    
    player_stats = player_stats.fillna(0.0)
    player_stats = player_stats.astype(np.float64)  
    stats = player_stats.values
    # nan_values = np.argwhere(np.isnan(stats))
    # print (nan_values)      
    stats = stats / stats.max(axis=0)
    # print (stats[0])
    # nan_values = np.argwhere(np.isnan(stats))
    # print (stats[0])
    # print (nan_values)      
    return name, stats 

def model_cosine_similarity(name, stats, player_position):
    y = 0.0
    pos = 0
    sample_size = len(stats)

    # score = cosine_similarity([stats[player_position]], [stats[player_position]])
    # x = int(score[0][0] * 10000) / 100.0
    # print (score[0][0])
    # print (x)
    # if x > y:
    #     y = x
    #     # pos = i
    #     print (y)

    # print (name[player_position], "has similar stats to :", name[i])
    # print (score[0][0])
    # print (type(score[0][0]))
    # print (type(sim))

    print ("Comparing Similarity Between All Players in the Dataset ... ")
    for i in range(sample_size):
        if i == player_position:
            continue
        else:    
            score = cosine_similarity([stats[player_position]], [stats[i]])
            x = int(score[0][0] * 10000) / 100.0
            # print (name[player_position], "comparing to : ", name[i]," Index : ",i, " Score: ", str(x))
            if x > y:
                y = x
                pos = i
    print ("NBA Prospect : ",name[player_position])
    print ("Closest Match : ", name[pos])
    print ("Cosine Similarity Score : ", str(y))
    print ("Player Index : ",pos)  

def player_vs_prospect_cosine_similarity(draft_prospect_name, draft_prospect_stats, nba_player_name, nba_player_stats, prospect_name):
    y = 0.0
    pos = 0
    sample_size = len(nba_player_stats)
    print ("Sample Size :", sample_size)
    
    index_array = np.where(draft_prospect_name== prospect_name)
    prospect_index = index_array[0][0]
    print (prospect_index)

    # score = cosine_similarity([draft_prospect_stats[prospect_index]], [nba_player_stats[0]])
    # x = int(score[0][0] * 10000) / 100.0
    # print (score[0][0])
    # print (x)
    # if x > y:
    #     y = x
    #     # pos = i
    #     print (y)

    # # print (name[player_position], "has similar stats to :", name[i])
    # # print (score[0][0])
    # # print (type(score[0][0]))
    # # print (type(sim))

    print ("Matching Prospect to NBA player... ")
    for i in range(500):   
        score = cosine_similarity([draft_prospect_stats[prospect_index]], [nba_player_stats[i]])
        x = int(score[0][0] * 10000) / 100.0
        print (draft_prospect_name[prospect_index], "comparing to : ", nba_player_name[i]," Index : ",i, " Score: ", str(x))
        if x > y:
            y = x
            pos = i
    print ("NBA Prospect : ",draft_prospect_name[prospect_index])
    print ("Closest Match : ", nba_player_name[pos])
    print ("Cosine Similarity Score : ", str(y))
    print ("NBA Player Index : ",pos)  

    print ("NBA Draft Prospect vs Top NBA Player")
    print (draft_prospect_stats[prospect_index])    
    print (nba_player_stats[pos])   

# https://stackoverflow.com/questions/48013402/knn-algorithm-that-return-2-or-more-nearest-neighbours
def model_knn(name, stats, player_position):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(stats) 
    distances, indices = nbrs.kneighbors(stats)
    print (indices[player_position])
    # source_pos = indices[player_position][0]
    target_pos = indices[player_position][1]
    print (distances[player_position])
    print (distances[target_pos])
    # print (source_pos)
    # print (target_pos)
    # print ("NBA Prospect : ",name[player_position])
    # print ("Closest Match : ", name[target_pos])
    # print ("Cosine Similarity Score : ", str(y))
    # print ("Player Index : ",pos)     

# filename = "Datasets/players_stats.csv"
# player_position = 29
# name, stats = load_data(filename)
# cosine_similarity_model(name, stats, player_position)
# model_knn(name, stats, player_position)

prospect_name = "Donovan Mitchell"
# filename = "Datasets/nba_draft_prospects.csv"
filename = "Datasets/2017Draft.csv"
draft_prospect_name, draft_prospect_stats = load_nba_data(filename)
# filename = "Datasets/top_nba_players.csv"
filename = "Datasets/collegeStats.csv"
nba_player_name, nba_player_stats = load_nba_data(filename)
player_vs_prospect_cosine_similarity(draft_prospect_name, draft_prospect_stats, nba_player_name, nba_player_stats,prospect_name)