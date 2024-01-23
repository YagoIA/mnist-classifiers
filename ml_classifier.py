from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
from numpy.core.arrayprint import format_float_scientific
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

clf1 = BaggingClassifier(svm.SVC(kernel='rbf', verbose=True), max_samples=0.4)
clf2 = RandomForestClassifier()
clf3 = AdaBoostClassifier()
clf4 = DecisionTreeClassifier()

#
clf5 = et_clf = ExtraTreesClassifier()
clf6 = HistGradientBoostingClassifier()

hard_voting_ensemble = VotingClassifier(estimators=[
   ("random_forest", clf2),
    # ("ada_boost", clf3),
	# ("decision tree", clf4)
	("HistGradientBoosting", clf6),
	("extraTrees", clf5),
    ("svm", clf1)

], voting="hard", verbose=True)

#classifiers = [clf1, clf2, clf3, clf4]
classifiers = [clf2]

#print(trainX.shape)
#print(trainX[0].reshape(1, -1))

reshaped_trainX = []
reshaped_testX = []

for x in trainX:
	reshaped_trainX.append(np.array(x).reshape(1, -1)[0])

for x in testX:
	reshaped_testX.append(np.array(x).reshape(1, -1)[0])

hard_voting_ensemble.fit(reshaped_trainX, trainy)

# for idx, classifier in enumerate(classifiers):
#   classifier.fit(reshaped_trainX, trainy)

# predicted = []

# for x, y in zip(testX, testy):
# 	predictions = []
# 	for classifier in classifiers:
# 		prediction = classifier.predict(np.array(x).reshape(1, -1))
# 		predictions.append(prediction[0])

# 	predicted.append(max(set(predictions), key=predictions.count))

predicted = hard_voting_ensemble.predict(reshaped_testX)
assert len(predicted) == len(testy)
print(accuracy_score(predicted, testy))