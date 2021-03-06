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
    # player_stats = player_stats.drop(player_stats[player_stats["Games Played"] < 50].index)
    # player_stats = df[['PTS','FGM']].copy()
    player_stats = player_stats.astype(np.float64)
    stats = player_stats.values
    stats = stats / stats.max(axis=0)
    return name, stats

def load_nba_data(filename):
    print ('Loading data...')
    df = pd.read_csv(filename,error_bad_lines=False) # ,error_bad_lines=False
    print ('Data has been loaded sucessfully')
    print ()
    player_name = df[['name']].copy()
    name = player_name.values
    # player_stats = df[['G','MP','FG','FGA','FGP','2P','2PA','2PP','3P','3PA','3PP','FT','FTA','FTP','TRB','AST','STL','BLK','TOV','PF','PTS']].copy()
    # player_stats = df[['g','mp','fg','fga','fg3','fg3a','ft','fta','orb','trb','ast','stl','blk','tov','pf','pts','fg_pct','fg3_pct','ft_pct','mp_per_g','pts_per_g','trb_per_g','ast_per_g']].copy()
    # player_stats = df[['fg_pct','fg3_pct','ft_pct','trb','ast','stl','blk','tov']].copy()  
    player_stats = df[['pts_per_g','trb_per_g','ast_per_g']].copy()  
    # print (player_stats.iloc[23])    
    player_stats = player_stats.fillna(0.0)
    player_stats = player_stats.astype(np.float64) 
    player_stats_val = player_stats.values 
    stats = player_stats.values
    # nan_values = np.argwhere(np.isnan(stats))
    # print (nan_values)      
    stats = stats / stats.max(axis=0)
    # print (stats[0])
    # nan_values = np.argwhere(np.isnan(stats))
    # print (stats[0])
    # print (nan_values)      
    return name, stats, player_stats_val 

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

def player_vs_prospect_cosine_similarity(draft_prospect_name, draft_prospect_stats, nba_player_name, nba_player_stats, prospect_name, dpsu, npsu):
    y = 0.0
    pos = 0
    sample_size = len(nba_player_stats)
    # print ("Sample Size :", sample_size)
    
    index_array = np.where(draft_prospect_name== prospect_name)
    prospect_index = index_array[0][0]
    # print (prospect_index)

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

    print ()
    print ("\t\t\t\t Draft Prospect : ", draft_prospect_name[prospect_index][0]) 
    print ("pts_per_g \t trb_per_g \t ast_per_g")
    print (round(dpsu[prospect_index][0],3), '\t\t', round(dpsu[prospect_index][1],3), '\t\t', round(dpsu[prospect_index][2],3))
    # print ("Closest Match : ", nba_player_name[pos])
    print ()    

    # print ("Matching Prospect to NBA player... ")
    for i in range(sample_size):   
        score = cosine_similarity([draft_prospect_stats[prospect_index]], [nba_player_stats[i]])
        x = int(score[0][0] * 10000) / 100.0
        # print (draft_prospect_name[prospect_index], "comparing to : ", nba_player_name[i]," Index : ",i, " Score: ", str(x))
        if x > 99.5:
            print ("\t\t\t\t Potential Player : ", nba_player_name[i][0])
            print ("pts_per_g \t trb_per_g \t ast_per_g")
            print (round(npsu[i][0],3), '\t\t', round(npsu[i][1],3), '\t\t', round(npsu[i][2],3))    
            print ()
            print ("Cosine Similarity Score : ", str(x))
            print ("NBA Player Index : ", pos + 2)  
            print ()                                
        if x > y:
            y = x
            pos = i
                    
    # print ("NBA Prospect : ",draft_prospect_name[prospect_index])
    # print ("NBA Prospect Stats :")

    # print ("\t\t\t\t Draft Prospect : ", draft_prospect_name[prospect_index][0]) 
    # print ("fg_pct \t\t fg3_pct \t ft_pct \t trb \t\t ast \t\t stl \t\t blk \t\t tov")
    # print (round(dpsu[prospect_index][0],3), '\t\t', round(dpsu[prospect_index][1],3), '\t\t', round(dpsu[prospect_index][2],3), '\t\t', dpsu[prospect_index][3], '\t\t', dpsu[prospect_index][4], '\t\t', dpsu[prospect_index][5], '\t\t', dpsu[prospect_index][6], '\t\t', dpsu[prospect_index][7])
    # # print ("Closest Match : ", nba_player_name[pos])
    # print ()
    # print ("\t\t\t\t Potential : ", nba_player_name[pos][0])
    # print ("fg_pct \t\t fg3_pct \t ft_pct \t trb \t\t ast \t\t stl \t\t blk \t\t tov")
    # print (round(npsu[pos][0],3), '\t\t', round(npsu[pos][1],3), '\t\t', round(npsu[pos][2],3), '\t\t', npsu[pos][3], '\t\t', npsu[pos][4], '\t\t', npsu[pos][5], '\t\t', npsu[pos][6], '\t\t', npsu[pos][7])    
    # print ()
    

    print ("\t\t\t\t Closest Match : ", nba_player_name[pos][0])
    print ("pts_per_g \t trb_per_g \t ast_per_g")
    print (round(npsu[pos][0],3), '\t\t', round(npsu[pos][1],3), '\t\t', round(npsu[pos][2],3))    
    print ()

    print ("Cosine Similarity Score : ", str(y))
    print ("NBA Player Index : ", pos + 2)  
    print ()
    print ("NBA Draft Prospect vs Top NBA Player")
    print (draft_prospect_stats[prospect_index])    
    print (nba_player_stats[pos])  

    print ()
    print ("Based on relative closeness to each field")

    find = 0
    lowerpts = round(dpsu[prospect_index][0],3) - 1.0
    upperpts = round(dpsu[prospect_index][0],3) + 1.0
    lowerrb = round(dpsu[prospect_index][1],3) - 1.0
    upperrb = round(dpsu[prospect_index][1],3) + 1.0
    lowerast = round(dpsu[prospect_index][2],3) - 1.0
    upperast = round(dpsu[prospect_index][2],3) + 1.0     
    
    while find == 0:    
        
        for i in range(sample_size):  
            if npsu[i][0] >= lowerpts and npsu[i][0] <= upperpts and npsu[i][1] >= lowerrb and npsu[i][1] <= upperrb and npsu[i][2] >= lowerast and npsu[i][2] <= upperast:
                print (nba_player_name[i][0])
                print (npsu[i])
                find = 1

        if find == 0:
            print ("Increasing Field of Search")
            lowerpts -=  1.0
            upperpts +=  1.0
            lowerrb -=  0.5
            upperrb += 0.5
            lowerast -=  0.5
            upperast +=  0.5 
    # print (df['two'] >= ) & (df['two'] < 0.5) 

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

prospect_name = "Kyle Kuzma"
# filename = "Datasets/nba_draft_prospects.csv"
filename = "Datasets/2017Draft.csv"
draft_prospect_name, draft_prospect_stats,  draft_prospect_stats_unnormalised = load_nba_data(filename)
# filename = "Datasets/top_nba_players.csv"
filename = "Datasets/collegeStats.csv"
nba_player_name, nba_player_stats, nba_player_stats_unnormalised = load_nba_data(filename)
player_vs_prospect_cosine_similarity(draft_prospect_name, draft_prospect_stats, nba_player_name, nba_player_stats, prospect_name, draft_prospect_stats_unnormalised, nba_player_stats_unnormalised)