from sklearn import svm, datasets
import pickle
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X,y)

#Save model
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0:1]))
print(y[0])

#joblib
from sklearn.externals import joblib
joblib.dump(clf, 'iris-model.pkl')
clf = joblib.load('iris-model.pkl')
print(list(clf.predict(X[0:150])))