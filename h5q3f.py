"""
Created on Thu Apr 14 21:27:02 2022

@author: ronal
"""
import seaborn as sns
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

ratings = pd.read_csv('ratings_small.csv')
reader = Reader(rating_scale = (0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# ratings = ratings.iloc[:500,:]

avg_rmse_u = []
avg_rmse_i = []
for x in range(20, 650, 20):
    print(x," Neighbors")
    print("User-Based Collaborative Filtering using MSD")
    sim_options = {'name':'MSD', 'user_based':True}
    user_cf_msd = KNNBasic(k=x,sim_options = sim_options, verbose = False)
    
    scores_msd_user = pd.DataFrame()
    scores_msd_user = cross_validate(user_cf_msd, data, measures = ['RMSE', 'MAE'], cv=5)
    print(scores_msd_user)
    avg_rmse_msd_u = scores_msd_user['test_rmse'].mean()

    avg_mae_msd_u = scores_msd_user['test_mae'].mean()
    avg_rmse_u.append(avg_rmse_msd_u)
    print()
    

    #Item based Collaborative Filtering - MSD
    print("Item-Based Collaborative Filtering using MSD")
    sim_options = {'name':'MSD', 'user_based':False}
    item_cf_msd = KNNBasic(k=x, sim_options = sim_options, verbose = False)
    
    scores_msd_item = pd.DataFrame()
    scores_msd_item = cross_validate(item_cf_msd, data, measures = ['RMSE', 'MAE'], cv=5)
    print(scores_msd_item)
    avg_rmse_msd_i = scores_msd_item['test_rmse'].mean()
    avg_mae_msd_i = scores_msd_item['test_mae'].mean()
    avg_rmse_i.append(avg_rmse_msd_i)
    print()
    
indvar = [x for x in range(20,650,20)]
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.plot(indvar, avg_rmse_u , '-ro')
ax1.grid(visible = True)
ax1.set_xlabel('K Nearest Neighbor Count')
ax1.set_ylabel('Mean RMSE Score')
ax1.set_title('User-Based CF')

   
ax2.plot(indvar,avg_rmse_i , '-ro')
ax2.set_xlabel('K Nearest Neighbor Count')
ax2.set_ylabel('Mean RMSE Score')
ax2.set_title('Item-Based CF')
ax2.grid(visible = True)

    
fig.tight_layout(pad=3)
# plt.figure(figsize = (5,5))
# x =  np.linspace(1,5,1)


