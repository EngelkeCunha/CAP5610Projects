# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:22:13 2022

@author: ronal
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m
import pandas as pd

def randomCentroid(data):
    centroids=[]
    random_cents=[np.random.randint(data.shape[0]) for i in range(10)]
    for point in random_cents:
        centroids.append(data[point])
    return np.asarray(centroids)

def predict(data):
    labels_act = []
    cl = data[['cluster', 'label']]
    y = np.bincount(cl['label'])
    cl = cl.sort_values("cluster")
    x = cl.value_counts().unstack()
    labels = []
    x= x.fillna(0)

    labels.append(x.idxmax(axis="columns"))
    
    columns_clusters = data['cluster']
    columns_clusters = np.array(columns_clusters)
    
    labels = np.array(labels)
    labels = labels.reshape(-1,1)
    x = np.array(x)    
    for i in range(data.shape[0]):
        point_cl = columns_clusters[i]
        label = labels[point_cl]
        labels_act.append(label)
      
    return labels_act

def clusterAssignment(X,centroids,k, z):
    clusters=[] 
    totalSSE= []
    if z == 0:
        for i in range(X.shape[0]):
            euc_dist=[]
            for j in range(k):
                euc_dist.append(np.sqrt(np.sum((X[i] - centroids[j]) ** 2)))
            argmin=np.argmin(euc_dist) 
            minDist = np.min(euc_dist)
            totalSSE.append(minDist)
            clusters.append(argmin) 
    elif z==1:
        for i in range(X.shape[0]):
            cos_dist=[]
            for j in range(k):
                cos_dist.append(1 - (np.dot(X[i],centroids[j]) / (np.sqrt(np.dot(X[i],X[i])) * np.sqrt(np.dot(centroids[j],centroids[j])))))
            argmin=np.argmin(cos_dist) 
            minDist = np.min(cos_dist)
            totalSSE.append(minDist)
            clusters.append(argmin) 
    elif z==2:
        for i in range(X.shape[0]):
            jacc_dist=[]
            for j in range(k):
                jacc_dist.append(1 - (np.sum(np.minimum(X[i],centroids[j])) / np.sum(np.maximum(X[i],centroids[j]))))
            argmin=np.argmin(jacc_dist)
            minDist = np.min(jacc_dist)
            totalSSE.append(minDist)
            clusters.append(argmin) 
   
    return np.asarray(clusters), np.asarray(totalSSE)


def centroidDelta(cent1, cent2):
    dimensionalDelta = []
    for dim in range(len(cent1)):
        dimensionalDelta.append(np.sqrt(np.sum((cent1[dim] - cent2[dim]) ** 2)))
    finalDelta = sum(dimensionalDelta)    
    
    return finalDelta

def centroidRecomputation(data,clusters,k):
    centroids = [] 
    for cent in range(k):
        singleCluster = []
        for dat in range(data.shape[0]):
            if clusters[dat]==cent:
                singleCluster.append(data[dat])
        centroids.append(np.mean(singleCluster,axis=0))
        
    return np.asarray(centroids)

def KMeans(data, labels, k, z):
    previousCentroids = randomCentroid(data)
    cluster = clusterAssignment(data,previousCentroids, k, z)
    initialCentroids = previousCentroids

    canImprove = True
    stopCriteria = True
    iterations = 1
    i = 0
    diff = 100
    SSE = []
     
    while(stopCriteria):
        clusters, roundSSE = clusterAssignment(data, previousCentroids, k , z)
        SSE.append(sum(roundSSE))
        newCentroids = centroidRecomputation(data,clusters,k)
        
        delta = centroidDelta(previousCentroids, newCentroids)
        print("centroid delta = ", round(delta,2) , "     iteration ", iterations, " SSE: ", round(SSE[i], 3))
        previousCentroids = newCentroids
        
        if (len(SSE) < 2):
            stopCriteria = True
        else:
            if (round(SSE[i], 3) > round(SSE[i - 1], 3)):
                print("\nINCREASE IN SSE! ABORTING!\n")
                stopCriteria= False
                dum_clusters, roundSSEinEuc = clusterAssignment(data, previousCentroids, k, 0)
                print("total SSE (in Euc): ", sum(roundSSEinEuc) )
                break
            elif (delta == 0):  
                print("\nCENTROID DELTA = 0 ABORTING!\n")
                stopCriteria = False
                dum_clusters, roundSSEinEuc = clusterAssignment(data, previousCentroids, k, 0)
                print("total SSE (in Euc): ", sum(roundSSEinEuc) )
                break
            if (iterations >= 100):
                print("\nEXCEEDED TOTAL ITERATION COUNT! ABORTING \n")
                stopCriteria= False
                dum_clusters, roundSSEinEuc = clusterAssignment(data, previousCentroids, k, 0)
                print("total SSE (in Euc): ", sum(roundSSEinEuc) )
                break
        i = i + 1
        iterations = iterations + 1
     
    data = pd.DataFrame(data)
    data['cluster'] = clusters
    data['label'] = labels
    
    prediction = predict(data)
    labels = np.array(labels)
    accurate = (prediction == labels)
    accuracy = accurate.sum() / accurate.size
    print("overall prediction accuracy: ", round(accuracy, 2))
    return clusters, previousCentroids, prediction

#--------------------------------------------------------------------------------------------------#
label = ['A']

data = pd.read_csv("data.csv", header = None ,skiprows=(0))
labels = pd.read_csv("label.csv", skiprows=(0), names = label)
data = np.array(data)
labels = np.array(labels)

k = 10

print("Euclidian-Based K Means: ")
clusters1, centroids1, prediction1 = KMeans(data, labels, k, 0 )

print()

print("Cosine Similarity-Based K Means: ")
clusters2, centroids2, prediction2 = KMeans(data, labels, k, 1)

print()

print('Jaccard Similarity-Based K Means: ')
clusters3, centroids3, prediction3 = KMeans(data, labels, k, 2)

