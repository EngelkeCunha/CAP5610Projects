# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:38:25 2022

@author: ronal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import math
from mpl_toolkits.mplot3d import Axes3D

columns = ['class','1','sqrt(2)*x','x^2']

data = pd.read_csv('task6.csv', skiprows=1, names=columns)
df = pd.DataFrame(data)

X = df.values[:,1:]
Y = df.values[:,0]

svc = svm.SVC(kernel = 'linear')
svc.fit(X,Y)

X0 = X[:,0]
X1 = X[:,1]
X2 = X[:,2]

z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

space = np.linspace(-2,2,2)
x, y = np.meshgrid(space,space)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')

ax.plot_surface(x, y, z(x,y))

