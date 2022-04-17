# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:55:14 2022

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

reader = Reader(rating_scale = (0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)


# PMF - Probabilistic Matrix Factorization 
print("Probabilistic Matrix Factorization")
svd = SVD(biased = False)
scores = pd.DataFrame
cross_validate(svd, data, measures = ['RMSE', 'MAE'], cv=5, verbose = True)
print(scores)
rmse_scores = scores['test_rmse']
mae_scores = scores['test_mae']

avg_rmse = rmse_scores.mean()
avg_mae = mae_scores.mean()

print()

#User based Collaborative Filtering - MSD
print("User-Based Collaborative Filtering using MSD")
sim_options = {'name':'MSD', 'user_based':True}
user_cf_msd = KNNBasic(sim_options = sim_options, verbose = False)

scores_msd_user = pd.DataFrame()
scores_msd_user = cross_validate(user_cf_msd, data, measures = ['RMSE', 'MAE'], cv=5)
print(scores_msd_user)
avg_rmse_msd_u = scores_msd_user['test_rmse'].mean()
avg_mae_msd_u = scores_msd_user['test_mae'].mean()
print()


#Item based Collaborative Filtering - MSD
print("Item-Based Collaborative Filtering using MSD")
sim_options = {'name':'MSD', 'user_based':False}
item_cf_msd = KNNBasic(sim_options = sim_options, verbose = False)


scores_msd_item = pd.DataFrame()
scores_msd_item = cross_validate(item_cf_msd, data, measures = ['RMSE', 'MAE'], cv=5)
print(scores_msd_item)
avg_rmse_msd_i = scores_msd_item['test_rmse'].mean()
avg_mae_msd_i = scores_msd_item['test_mae'].mean()

print()

#User based Collaborative Filtering - Cosine
print("User-Based Collaborative Filtering using Cosine")
sim_options = {'name':'cosine', 'user_based':True}
user_cf_cos = KNNBasic(sim_options = sim_options, verbose = False)


scores_cos_user = pd.DataFrame()
scores_cos_user = cross_validate(user_cf_cos, data, measures = ['RMSE', 'MAE'], cv=5)
print(scores_cos_user)
avg_rmse_cos_u = scores_cos_user['test_rmse'].mean()
avg_mae_cos_u = scores_cos_user['test_mae'].mean()

print()

#Item based Collaborative Filtering - Cosine
print("Item-Based Collaborative Filtering using Cosine")
sim_options = {'name':'cosine', 'user_based':False}
item_cf_cos = KNNBasic(sim_options = sim_options, verbose = False)

scores_cos_item = pd.DataFrame()
scores_cos_item = cross_validate(item_cf_cos, data, measures = ['RMSE', 'MAE'], cv=5)
print(scores_cos_item)
avg_rmse_cos_i = scores_cos_item['test_rmse'].mean()
avg_mae_cos_i = scores_cos_item['test_mae'].mean()

print()



#User based Collaborative Filtering - Pearson
print("User-Based Collaborative Filtering using Pearson")
sim_options = {'name':'pearson', 'user_based':True}
user_cf_pear = KNNBasic(sim_options = sim_options, verbose = False)


scores_pear_user = pd.DataFrame()
scores_pear_user = cross_validate(user_cf_pear, data, measures = ['RMSE', 'MAE'], cv=5)
print(scores_pear_user)
avg_rmse_pear_u = scores_pear_user['test_rmse'].mean()
avg_mae_pear_u = scores_pear_user['test_mae'].mean()

print()

#Item based Collaborative Filtering - Pearson
print("Item-Based Collaborative Filtering using Pearson")
sim_options = {'name':'pearson', 'user_based':False}
item_cf_pear = KNNBasic(sim_options = sim_options, verbose = False)


scores_pear_item = pd.DataFrame()
scores_pear_item = cross_validate(item_cf_pear, data, measures = ['RMSE', 'MAE'], cv=5)
print(scores_pear_item)
avg_rmse_pear_i = scores_pear_item['test_rmse'].mean()
avg_mae_pear_i = scores_pear_item['test_mae'].mean()


print()



Urmse = [avg_rmse_msd_u, avg_rmse_cos_u, avg_rmse_pear_u]
Irmse = [avg_rmse_msd_i, avg_rmse_cos_i, avg_rmse_pear_i]
tot = Irmse + Urmse
avgs_rmse = pd.DataFrame({"User-Based":[avg_rmse_msd_u, avg_rmse_cos_u, avg_rmse_pear_u],
"Item-Based":[avg_rmse_msd_i, avg_rmse_cos_i, avg_rmse_pear_i]}, 
index = ["MSD", "Cosine", "Pearson"])


r1 = np.arange(len(avgs_rmse))
r2 = [x + 0.25 for x in r1]
plt.figure(figsize=(5,5))
plt.bar(r1, Irmse, width = 0.25, label = "Item-Based")
plt.bar(r2, Urmse, width = 0.25, label = "User-Based")
plt.xlabel('Similarity Measurement', fontweight='bold')
plt.ylabel('Mean RMSE', fontweight='bold')
plt.title('Mean RMSE Scores', fontweight='bold')
plt.xticks([r+0.125 for r in range(len(Urmse))], ['MSD', 'Cosine', 'Pearson'])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
plt.grid()

plt.legend()

print(avgs_rmse)
print()

Umae = [avg_mae_msd_u, avg_mae_cos_u, avg_mae_pear_u]
Imae = [avg_mae_msd_i, avg_mae_cos_i, avg_mae_pear_i]
tot = Imae + Umae
avgs_mae = pd.DataFrame({"User-Based":[avg_mae_msd_u, avg_mae_cos_u, avg_mae_pear_u],
"Item-Based":[avg_mae_msd_i, avg_mae_cos_i, avg_mae_pear_i]}, 
index = ["MSD", "Cosine", "Pearson"])

r1 = np.arange(len(avgs_mae))
r2 = [x + 0.25 for x in r1]
plt.figure(figsize=(5,5))
plt.bar(r1, Imae, width = 0.25, label = "Item-Based")
plt.bar(r2, Umae, width = 0.25, label = "User-Based")
plt.xlabel('Similarity Measurement', fontweight='bold')
plt.ylabel('Mean MAE', fontweight='bold')
plt.title('Mean MAE Scores', fontweight='bold')
plt.xticks([r+0.125 for r in range(len(Umae))], ['MSD', 'Cosine', 'Pearson'])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])

plt.grid()
plt.legend()

print(avgs_mae)

# avgs_rmse.plot(kind = "bar", figsize=(15,10))
# avgs_mae.plot(kind = "bar", figsize=(15,10))
# fig1.title("Average RMSE - Item + User-based Collaborative Filtering")
# fig2.title("Average MAE - Item + User-based Collaborative Filtering")
# fig1.xlabel("Similarity Measurement")
# fig2.xlabel("Similarity Measurement")


