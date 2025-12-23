import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

titanic_data = pd.read_csv(r"D:\Internship\titanic.csv")

titanic_data = titanic_data.dropna(subset=['Survived'])

X = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

import seaborn as sns
plt.figure()
plt.subplot(2,3,1)
sns.countplot(x = titanic_data['Pclass'], hue = titanic_data['Survived'], palette = 'Pastel1')
plt.subplot(2,3,2)
sns.countplot(x = titanic_data['Sex'], hue = titanic_data['Survived'], palette = 'Pastel2')
plt.subplot(2,3,3)
sns.countplot(x = titanic_data['Age'], hue = titanic_data['Survived'], palette = 'Spectral')
plt.subplot(2,3,4)
sns.countplot(x = titanic_data['SibSp'], hue = titanic_data['Survived'], palette = 'BrBG')
plt.subplot(2,3,5)
sns.countplot(x = titanic_data['Parch'], hue = titanic_data['Survived'], palette = 'BuPu')
plt.subplot(2,3,6)
sns.countplot(x = titanic_data['Fare'], hue = titanic_data['Survived'], palette = 'BuGn')
plt.show()

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

X.loc[:, 'Sex'] = X['Sex'].map({'female': 0, 'male': 1})

X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier()

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(accuracy)
print(classification_rep)

from sklearn import tree

plt.figure()
plt.subplot(3, 3, 1)
tree.plot_tree(rf_classifier.estimators_[0], feature_names = titanic_data.columns, filled = True, rounded = True);

plt.subplot(3, 3, 2)
tree.plot_tree(rf_classifier.estimators_[12], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 3)
tree.plot_tree(rf_classifier.estimators_[24], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 4)
tree.plot_tree(rf_classifier.estimators_[36], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 5)
tree.plot_tree(rf_classifier.estimators_[48], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 6)
tree.plot_tree(rf_classifier.estimators_[60], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 7)
tree.plot_tree(rf_classifier.estimators_[72], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 8)
tree.plot_tree(rf_classifier.estimators_[84], feature_names = titanic_data.columns, filled = True);

plt.subplot(3, 3, 9)
tree.plot_tree(rf_classifier.estimators_[99], feature_names = titanic_data.columns, filled = True);
plt.show()
