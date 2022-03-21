# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:20:47 2022

@author: ronal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets


columns = ['class','x1','x2']

data = pd.read_csv('task5.csv', skiprows=1, names=columns)
df = pd.DataFrame(data)

X = df.values[:,1:]
Y = df.values[:,0]

svc = svm.SVC(kernel='linear', C=3).fit(X,Y)

plt.scatter(X[:,0], X[:,1], c=Y, s=50)

ax = plt.gca()
xlimit = ax.get_xlim()
ylimit = ax.get_ylim()

xx = np.linspace(0, xlimit[1], 30)
yy = np.linspace(0, ylimit[1], 30)

YY, XX = np.meshgrid(yy, xx)
XY = np.vstack([XX.ravel(), YY.ravel()]).T
P = svc.decision_function(XY).reshape(XX.shape)

ax.contour(XX,YY,P,colors="black", levels=[-1,0,1],alpha = 1, linestyles=["--", "-", "--"])
ax.scatter(svc.support_vectors_[:,0], svc.support_vectors_[:,1], s=300, 
            linewidth=2, facecolors="none", edgecolors="black")

