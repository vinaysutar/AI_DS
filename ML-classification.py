import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
'''from sklearn.metrics import confusion_matrix

logr=LogisticRegression()
df = pd.read_csv("C:\\Users\\admin\\Documents\\iris3\\Iris.csv")
x =df.drop("Id",axis=1)
x = x.drop('Species',axis=1)
y = df['Species']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.3)
logr.fit(X_train,Y_train)
y_pred=logr.predict(X_test)
score=accuracy_score(Y_test,y_pred)
print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
print(confusion_matrix(Y_test,y_pred))'''
'''from sklearn.naive_bayes import GaussianNB
df = pd.read_csv("C:\\Users\\admin\\Documents\\iris3\\Iris.csv")
nb=GaussianNB()
x =df.drop("Id",axis=1)
x = x.drop('Species',axis=1)
y = df['Species']
X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.3)
nb.fit(X_train,Y_train)
y_pred=nb.predict(X_test)
score=accuracy_score(Y_test,y_pred)

print("Naive bayes:",accuracy_score(Y_test,y_pred))'''
'''from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:\\Users\\admin\\Documents\\iris3\\Iris.csv")
kN = KNeighborsClassifier()
x = df.drop("Id", axis=1)
x = x.drop('Species', axis=1)
y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=1, test_size=0.3)
kN.fit(X_train, Y_train)
y_pred = kN.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print(score)'''

'''from sklearn import tree


df = pd.read_csv("C:\\Users\\admin\\Documents\\iris3\\Iris.csv")
dt = tree.DecisionTreeClassifier()
x = df.drop("Id", axis=1)
x = x.drop('Species', axis=1)
y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=1, test_size=0.3)
dt.fit(X_train, Y_train)
y_pred = dt.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print(score)'''
'''from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()
df = pd.read_csv("C:/Users/admin/Documents/iris3/Iris.csv")
x = df.drop("Id", axis=1)
x = x.drop('Species', axis=1)
y = df['Species']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=1, test_size=0.3)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)
score = accuracy_score(Y_test, y_pred)
print(score)
'''