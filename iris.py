'''import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/admin/Desktop/iris 2/Iris.csv")'''
'''import pandas as pd'''

'''
# print(df)
# print(df.head())
# print(df.tail())
print(df.shape)
print(df.loc[df["Species"] == "Iris-setosa"])
print(df["Species"].value_counts())
print("median", df["SepalLengthCm"].median())
print("mean", df["SepalLengthCm"].mean())
print("sum", df["SepalLengthCm"].sum())'''
'''plt.scatter(df["PetalLengthCm"],df["PetalWidthCm"])
plt.title("Scatter Plot")
plt.xlabel("Petal Lenght")
plt.ylabel("Petal width")
plt.show()'''
'''plt.hist(df["SepalWidthCm"])
plt.title("Scatter Plot")
plt.xlabel("Sepal Lenght")
plt.ylabel("Sepal Width")
plt.show()'''
'''
sns.barplot(df["Species"])
sns.barplot(df["PetalLengthCm"])
plt.title("Bar Plot")
plt.show()'''
'''mylable = ["Iris-setosa", "Iris-vericolor", "Iris-variginica"]
sizes = [50, 50, 50]
plt.pie(sizes, labels=mylable, explode=(0.1, 0.1, 0.1))
plt.axis('equal'
plt.show()
'''
'''sns.countplot(x='SepalWidthCm', data = df) 
plt.title("count PLot For Species")
plt.show()'''
'''sns.boxplot(x='SepalLengthCm', data=df)
plt.title("Box plot showing the distribution of length")
plt.show()'''

'''sns.heatmap(df.corr()) # if any column is string it cannot convert it to float or integer
plt.show()'''
'''
print(df.isnull().sum())
df['SepalWidthCm'].fillna((df['SepalWidthCm'].mean()), inplace=True)
df['Species'].fillna('mode', inplace=True)

df.replace('?', 'nun')
print(df.isnull().sum())
'''
'''print(df)
print(df['SepalLengthCm'])
Q1 = df['SepalLengthCm'].quantile(0.25)
Q3 = df['SepalLengthCm'].quantile(0.75)
IOR = Q3 - Q1
print(IOR)
upper = Q3 + 1.5 * IOR
lower = Q1 - 1.5 * IOR
print(upper)
print(lower)
out1 = df[df['SepalLengthCm'] < lower].values
out2 = df[df['SepalLengthCm'] > upper].values
df['SepalLengthCm'].replace(out1, lower, inplace=True)
df['SepalLengthCm'].replace(out2, upper, inplace=True)
T
sns.boxplot(x='SepalLengthCm', data=df)
plt.title("Box plot showing the distribution of length")
plt.show()'''
'''import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
reg=LinearRegression()
df=pd.read_csv("C:/Users/admin/OneDrive/Documents/HousingData.csv")
print(df.isnull().sum())
df['CRIM'].fillna((df['CRIM'].mean()),inplace=True)
df['ZN'].fillna((df['ZN'].mean()),inplace=True)
df['INDUS'].fillna((df['INDUS'].mean()),inplace=True)
df['CHAS'].fillna((df['CHAS'].mean()),inplace=True)
df['AGE'].fillna((df['AGE'].mean()),inplace=True)
df['LSTAT'].fillna((df['LSTAT'].mean()),inplace=True)
print(df.isnull().sum())
df['LSTAT'].fillna((df['LSTAT'].mean()), inplace=True)
df['LSTAT'].fillna('mode', inplace=True)

df.replace('?', 'NA')
print(df.isnull().sum())
x=df.drop('MEDV', axis =1)

y=df['MEDV']




print(x)
print(y)

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.1)
train=reg.fit(X_train,Y_train)
y_pred=reg.predict(X_test)
score=mean_squared_error(Y_test,y_pred)
print(score)
'''
'''#inear regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns


df=pd.read_csv("C:/Users/admin/OneDrive/Documents/HousingData.csv")
print(df)
df.replace('?',"OM")
print(df.isnull().sum())

df['CRIM']=pd.cut(df['CRIM'],3,labels=['0','1','2'])
df['ZN']=pd.cut(df['ZN'],3,labels=['0','1','2'])
df['INDUS']=pd.cut(df['INDUS'],3,labels=['0','1','2'])
df['CHAS']=pd.cut(df['CHAS'],3,labels=['0','1','2'])
print(df)

X = df.drop('INDUS', axis =1)
X = X.drop('CHAS', axis=1)
Y = df['CHAS']
print(Y)
le=LabelEncoder()
le.fit(Y)
Y = le.transform(Y)
print(Y)

sns.boxplot(y='CRIM', data = df)                #x=horizontal and  y=vertical
plt.title("Box plot showing the distribution of sepal length")
plt.show()


Q1 = df['CRIM'] .quantile(0.25)
Q3 = df['CRIM'] .quantile(0.75)

IQR = Q3 - Q1
print(IQR)

upper = Q3 + 1.5*IQR
lower = Q1 - 1.5*IQR

print(lower)
print(upper)

out1=df[df['CRIM'] < lower].values
out2=df[df['CRIM'] > upper].values

df['CRIM'].replace(out1,lower)
df['CRIM'].replace(out2,lower)

sns.boxplot(df["CRIM"])
plt.show()'''