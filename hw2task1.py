import pandas as pd
import numpy as np
from sklearn import preprocessing, tree, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
import graphviz
from sklearn.ensemble import RandomForestClassifier

def preprocess_training_data_dt(data):
   
    X = data[columns]
    Y = data.Survived
    X = X.drop(X.columns[1], axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    imputer.fit(X['Age'].values.reshape(-1,1))
    X['Age'] = imputer.transform(X['Age'].values.reshape(-1,1))
    
    #dropping name, ticket, cabin
    X = X.drop(X.columns[[2,7,9]], axis=1)  
    
    #encoding sex, pclass, embarked
    X = pd.get_dummies(X, columns=['Sex'])  
    X = pd.get_dummies(X, columns=['Pclass'])
    X = pd.get_dummies(X, columns=['Embarked'])

    X = feature_selection(X,Y)
   
    return X, Y

def preprocess_testing_data_dt(X_train, data1, data2):
    
    X = data1[columns2]
    Y = data2[columns3]
    Y = Y.drop(Y.columns[[0]], axis=1)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    imputer.fit(X['Age'].values.reshape(-1,1))
    X['Age'] = imputer.transform(X['Age'].values.reshape(-1,1))
    
    imputer.fit(X['Fare'].values.reshape(-1,1))
    X['Fare'] = imputer.transform(X['Fare'].values.reshape(-1,1))
    
    X = X.drop(X.columns[[2,7,9]], axis=1)  
    X = pd.get_dummies(X, columns=['Sex'])
    X = pd.get_dummies(X, columns=['Pclass'])
    X = pd.get_dummies(X, columns=['Embarked'])

    X = X[X_train.columns]
    return X , Y

def preprocess_training_data_rf(data):
    
    X = data[columns]
    Y = data.Survived
    X = X.drop(X.columns[1], axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X['Age'].values.reshape(-1,1))
    X['Age'] = imputer.transform(X['Age'].values.reshape(-1,1))

    #dropping name, ticket, cabin
    X = X.drop(X.columns[[2,7,9]], axis=1)  
    
    #encoding sex, pclass, embarked
    X = pd.get_dummies(X, columns=['Sex'])  
    X = pd.get_dummies(X, columns=['Pclass'])
    X = pd.get_dummies(X, columns=['Embarked'])
    
    X = feature_selection(X,Y)
    
    return X, Y

def preprocess_testing_data_rf(X_train_rf, data1, data2):
    
    X = data1[columns2]
    Y = data2[columns3]
    Y = Y.drop(Y.columns[[0]], axis=1)
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    
    imputer.fit(X['Age'].values.reshape(-1,1))
    X['Age'] = imputer.transform(X['Age'].values.reshape(-1,1))
    
    imputer.fit(X['Fare'].values.reshape(-1,1))
    X['Fare'] = imputer.transform(X['Fare'].values.reshape(-1,1))
    
    X = X.drop(X.columns[[2,7,9]], axis=1)  
    X = pd.get_dummies(X, columns=['Sex'])
    X = pd.get_dummies(X, columns=['Pclass'])
    X = pd.get_dummies(X, columns=['Embarked'])

    X = X[X_train_rf.columns]
    
    
    return X, Y

def feature_selection(X, Y):

    chi2_values = chi2(X,Y)
    # results2 = pd.Series(chi2_values[1], index = X.columns)
    # results2.sort_values(ascending = False, inplace = True)
    # results2.plot.bar()
    
    # results1 = pd.Series(chi2_values[0], index = X.columns)
    # results1.sort_values(ascending = False, inplace = True)
    # results1.plot.bar()
    
    X_new = SelectKBest(chi2,k=5).fit(X,Y)

    best_features = []


    for bool, feature in zip(X_new.get_support(), X.columns):
        if bool:
            best_features.append(feature)
    
    X = SelectKBest(chi2,k=5).fit_transform(X,Y)
    
    X_final = pd.DataFrame(data = X, columns = best_features)
    
    print('Chi2 Best X_Train Features: {}\n'.format(best_features))
    
    return X_final
                      

def build_DT(X_train, Y_train, X_test):
    
    dt = tree.DecisionTreeClassifier(splitter="best") #29 ~82% accuracywas best
    dt = dt.fit(X_train, Y_train)
    
    prediction_array = dt.predict(X_test)
    
    return prediction_array, dt

def cross_validation(clf, X, Y):
    
    scores = cross_val_score(clf, X, Y, cv=5)
    
    return scores

def run_1000_accuracy(tree, X_test, Y_actual):
   
    avg = 0
    for i in range(1,1001):
        Y_pred = tree.predict(X_test)
        run_avg = metrics.accuracy_score(Y_actual, Y_pred)
        avg = (avg + run_avg)/i
        i+=1
        
    return (avg*1000)    

def build_Graph(dt, X):
    dt_export = tree.export_graphviz(dt, out_file=None,feature_names=X.columns.tolist(), class_names =  ['Survived=0', 'Survived=1'], filled = True, rounded = True)

    graph = graphviz.Source(dt_export)
    graph.render("dt_topic1")
    tree.plot_tree(dt,feature_names=X.columns.tolist(), class_names= ['Survived=0', 'Survived=1'] ,filled = True, rounded = True)
    
    return

def build_RF(X_train, Y_train, X_test):
    
    rf = RandomForestClassifier(n_estimators=9)
    rf = rf.fit(X_train, Y_train)
    
    prediction_array = rf.predict(X_test)
    
    return prediction_array, rf
########################################################################

columns = ['PassengerID','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
columns2 = ['PassengerID','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
columns3 = ['Passenger','Survived']

df_train = pd.read_csv('train.csv', skiprows=(1) ,names = columns)
df_test1 = pd.read_csv('test.csv',skiprows= (1), names = columns2)
df_test2 = pd.read_csv('gender_submission.csv', skiprows=(1), names=columns3)

print('Decision Tree: \n')

X_train , Y_train = preprocess_training_data_dt(df_train)

X_test, Y_actual = preprocess_testing_data_dt(X_train, df_test1, df_test2)

Y_actual = Y_actual.values
Y_pred1 , dt = (build_DT(X_train, Y_train, X_test))

scores = cross_validation(dt, X_train, Y_train)


print("5-Fold Scores:", scores)

print("Cross-Validation AVG Accuracy = %0.4f w/ a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("One Run Accuracy = ", dt.score(X_test,Y_actual))
print("1000 Run Accuracy = ", run_1000_accuracy(dt, X_test, Y_actual),'\n')
build_Graph(dt, X_test)

#Now for Random Forest

print('Random Forest: \n')
X_train_rf, Y_train_rf = preprocess_training_data_rf(df_train)
X_test_rf, Y_actual_rf = preprocess_testing_data_rf(X_train_rf, df_test1, df_test2)

Y_pred2, rf = build_RF(X_train_rf, Y_train_rf, X_test_rf)

scores2 = cross_validation(rf, X_train_rf, Y_train_rf)

print("5-Fold Scores:", scores2)
print("Cross-Validation AVG Accuracy = %0.4f w/ a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))

print("One Run Accuracy = ", rf.score(X_test_rf, Y_actual))
print("1000 Run Accuracy = ", run_1000_accuracy(rf, X_test_rf, Y_actual),'\n')
