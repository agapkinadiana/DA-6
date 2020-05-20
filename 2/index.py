import pandas as pds
import mglearn
import matplotlib.pyplot as plt
import numpy as np


"In[10]:"
from sklearn.datasets import load_iris
iris_dataset = load_iris()

"In[21]:"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)

"In[22]:"
print("форма массива X_train: {}".format(X_train.shape))
print("форма массива y_train: {}".format(y_train.shape))

"In[23]:"
print("форма массива X_test: {}".format(X_test.shape))
print("форма массива y_test: {}".format(y_test.shape))

iris_dataframe = pds.DataFrame(X_train, columns=iris_dataset.feature_names)

from pandas.plotting import scatter_matrix

grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

#plt.show()

print("_________________ LAB2 _________________")

"In[25]:"
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

"In[26]:"
knn.fit(X_train, y_train)

"In[27]:"
X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))

"In[28]:"
prediction = knn.predict(X_new)
print("Прогноз: {}".format(prediction))
print("Спрогнозированная метка: {}".format(iris_dataset['target_names'][prediction]))

"In[29]:"
y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))

"In[30]:"
print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))

"In[31]:"
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

"In[32]:"
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))


