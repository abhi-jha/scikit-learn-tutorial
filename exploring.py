import numpy as np
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10,2000)
X = np.array(X, dtype='float32')
print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)

from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
clf.fit(iris.data, iris.target)

print(list(clf.predict(iris.data[:3])))

clf.fit(iris.data, iris.target_names[iris.target])

print(list(clf.predict(iris.data[:3])))



#Refitting and updating hyperparameters
#default is rbf

rng = np.random.RandomState(0)
X = rng.rand(100,10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5,10)

clf = SVC()
clf.set_params(kernel='linear').fit(X, y)
print(clf.predict(X_test))

clf.set_params(kernel='rbf').fit(X,y)
print(clf.predict(X_test))

#multiclass vs multilabel


from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]

y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
print(classif.fit(X,y).predict(X))

y = LabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X)) #output columns are the three labels 0, 1, 2 whose presence or absence corresponsds to y #########each row represent an instance from y

from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)

print(classif.fit(X,y).predict(X))#each row represent an instance from y
