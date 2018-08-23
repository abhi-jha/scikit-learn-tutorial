from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()

digits = datasets.load_digits()

classification = svm.SVC(gamma=0.001,C=100, verbose=True, random_state=43)

#train
classification.fit(digits.data[:-1], digits.target[:-1])

#predict
print(list(classification.predict(digits.data[-1:])))

