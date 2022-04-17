# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:57:06 2022

@author: ronal
"""
import pandas as pd
import numpy as np
from surprise import SVD, NMF, KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import accuracy
from surprise.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

ratings = pd.read_csv('ratings_small.csv')
movies = pd.read_csv('movies_metadata.csv')
# ratings = ratings.iloc[:500,:]

cross_iterator = KFold(n_splits = 5)

best_user_rmse = 200
best_item_rmse = 200
best_item_k = 0
best_user_k = 0

k = []
rmseI = []
rmseU = []
for x in range(1,125,1):

    reader = Reader(rating_scale = (0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    #User based Collaborative Filtering - MSD
    print("User-Based Collaborative Filtering using MSD")
    sim_options = {'name':'MSD', 'user_based':True}
    user_cf_msd = KNNBasic(k = x,sim_options = sim_options, verbose = False)
    
    scores_msd_user = pd.DataFrame()
    scores_msd_user = cross_validate(user_cf_msd, data, measures = ['RMSE', 'MAE'], cv=5)
    print(scores_msd_user)
    avg_rmse_msd_u = scores_msd_user['test_rmse'].mean()
    
    rmseU.append(avg_rmse_msd_u)
                    
    if avg_rmse_msd_u < best_user_rmse:
        best_user_rmse = avg_rmse_msd_u
        best_user_k = x
        
    avg_mae_msd_u = scores_msd_user['test_mae'].mean()
    print()
    
    
    #Item based Collaborative Filtering - MSD
    print("Item-Based Collaborative Filtering using MSD")
    sim_options = {'name':'MSD', 'user_based':False}
    item_cf_msd = KNNBasic(k = x, sim_options = sim_options, verbose = False)
    
    
    scores_msd_item = pd.DataFrame()
    scores_msd_item = cross_validate(item_cf_msd, data, measures = ['RMSE', 'MAE'], cv=5)
    print(scores_msd_item)
    avg_rmse_msd_i = scores_msd_item['test_rmse'].mean()
    
    rmseI.append(avg_rmse_msd_i)
    
    if avg_rmse_msd_i < best_item_rmse:
        best_item_rmse = avg_rmse_msd_i
        best_item_k = x
        
    avg_mae_msd_i = scores_msd_item['test_mae'].mean()
    
    k.append(x)
    print()
 
avgIrmse = pd.DataFrame(columns=['K Count', 'Avg RMSE'])
avgUrmse = pd.DataFrame(columns=['K Count', 'Avg RMSE'])    
 
avgIrmse['K Count'] = k
avgIrmse['Avg RMSE'] = rmseI

avgUrmse['K Count'] = k
avgUrmse['Avg RMSE'] = rmseU

print("User-Based Collaborative Filtering using MSD")
print("Best K Value: ", best_user_k)
print("With a best avg RMSE score of ", best_user_rmse)
print()

print("Item-Based Collaborative Filtering using MSD")
print("Best K Value: ", best_item_k)
print("With a best avg RMSE score of ", best_item_rmse)
print()