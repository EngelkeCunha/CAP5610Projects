# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:18:17 2022

@author: ronal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
from sklearn.svm import SVC

        
columns1 = ['x1', 'x2', 'y']

total = pd.read_csv('task2.csv', skiprows=(1), names=columns1)
df = pd.DataFrame(total)

X = df.values[:, :2]
Y = df.values[:, 2]


clf = SVC(kernel='linear', C=5)
clf.fit(X,Y)

plt.scatter(X[:,0], X[:,1], c=Y, s=30)
ax = plt.gca()
xlimit = ax.get_xlim()
ylimit = ax.get_ylim()

xx = np.linspace(xlimit[0], xlimit[1], 50)
yy = np.linspace(ylimit[0], ylimit[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
P = clf.decision_function(xy).reshape(XX.shape)

ax.contour(
    XX, YY, P, colors="k", levels=[-1, 1], alpha=1, linestyles=["--", "-", "--"]
)

ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.xlabel('X1', size=12)
plt.ylabel('X1*X2', size =12)
plt.title('Task 2 Decision Region Boundary', size = 12)


