import sys
import pandas as pds
import mglearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn

print("версия Python: {}".format(sys.version))
print("версия pandas: {}".format(pds.__version__))
print("версия matplotlib: {}".format(matplotlib.__version__))
print("версия NumPy: {}".format(np.__version__))
print("версия SciPy: {}".format(sp.__version__))
print("версия IPython: {}".format(IPython.__version__))
print("версия scikit-learn: {}".format(sklearn.__version__))

"In[10]:"
from sklearn.datasets import load_iris
iris_dataset = load_iris()

"In[11]:"
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

"In[12]:"
print(iris_dataset['DESCR'][:193] + "\n...")

"In[13]:"
print("Названия ответов: {}".format(iris_dataset['target_names']))

"In[14]:"
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))

"In[15]:"
print("Тип массива data: {}".format(type(iris_dataset['data'])))

"In[16]:"
print("Форма массива data: {}".format(iris_dataset['data'].shape))

"In[17]:"
print("Первые пять строк массива data:\n{}".format(iris_dataset['data'][:5]))

"In[18]:"
print("Тип массива target: {}".format(type(iris_dataset['target'])))

"In[19]:"
print("Форма массива target: {}".format(iris_dataset['target'].shape))

"In[20]:"
print("Ответы:\n{}".format(iris_dataset['target']))

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

plt.show()

