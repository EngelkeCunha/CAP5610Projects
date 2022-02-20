import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

columns =  ['Early','Finished hmk','Senior','Likes Coffee','Liked The Last Jedi','A']
X_columns = ['Early','Finished hmk','Senior','Likes Coffee','Liked The Last Jedi']
Y_columns = ['A']

sda = "S","F"
df = pd.read_csv("hw2topic5.csv",skiprows=(1),names=columns)
X = df[X_columns]
Y = df[['A']]

clf1 = tree.DecisionTreeClassifier(criterion="entropy",splitter = "best",max_depth=1)
clf1.fit(X,Y)
clf1.predict(X)

dt1_export = tree.export_graphviz(clf1, out_file=None,feature_names=X.columns.tolist(),class_names= ['No A', 'A'] ,filled = True, rounded = True)
                                  
graph = graphviz.Source(dt1_export)
graph.render("dt1_topic5")
tree.plot_tree(clf1,feature_names=X.columns.tolist(),class_names= ['No A', 'A'] ,filled = True, rounded = True)


#######

clf2 = tree.DecisionTreeClassifier(criterion="entropy",splitter = "best",max_depth=2)
clf2 = clf2.fit(X,Y)
clf2.predict(X)

dt2_export = tree.export_graphviz(clf2, out_file=None,feature_names=X.columns.tolist(), class_names =  ['No A', 'A'], filled = True, rounded = True)

graph = graphviz.Source(dt2_export)
graph.render("dt2_topic5")
tree.plot_tree(clf2,feature_names=X.columns.tolist(), class_names= ['No A', 'A'] ,filled = True, rounded = True)
