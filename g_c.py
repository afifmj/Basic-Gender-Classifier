from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.svm import SVC

clf_tree = tree.DecisionTreeClassifier()
clf_nb = GaussianNB()
clf_svc = SVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

l = input("Enter the height, weight and shoe size of the male/female: ").split()

# # # SVC Classifier # # #
clf_svc = clf_svc.fit(X,Y)
prediction = clf_svc.predict([l])
print("The SVC Classifier results are: "+ str(prediction))
# # # END # # #

# # # Decision tree Classifier # # #
clf_tree = clf_tree.fit(X,Y)
prediction = clf_tree.predict([l])
print("The Decision tree Classifier results are: "+ str(prediction))
# # # END # # #

