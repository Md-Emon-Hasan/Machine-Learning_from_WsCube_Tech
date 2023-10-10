# find decision tree with enrtopy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd

data = pd.read_csv('tennis.csv')
outlook = LabelEncoder()
humidity = LabelEncoder()
windy = LabelEncoder()
play = LabelEncoder()

data['outlook'] = outlook.fit_transform(data['outlook'])
data['humidity'] = outlook.fit_transform(data['humidity'])
data['windy'] = outlook.fit_transform(data['windy'])
data['play'] = outlook.fit_transform(data['play'])

features_cols = ['outlook','humidity','windy']
x = data[features_cols]
y = data.play

# training the data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# decision tree
classifier = DecisionTreeClassifier(criterion='gini')

# train the data
classifier.fit(x_train,y_train)

# predict a value
classifier.predict(x_test)
x_test # 0->No, 1->Yes, outlook[sunny->2,overcast->0,rainy->1]

# check predict score
classifier.score(x_test,y_test)

# visulize the graph
tree.plot_tree(classifier)
