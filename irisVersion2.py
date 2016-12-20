# import iris dataset from sklearn
from sklearn import datasets

iris = datasets.load_iris()

# seperate th features into the variable x and the labels into Y
# we do this because we can think of a classifier as a function f(x) = y
x = iris.data
y = iris.target

# Partition the dataset into a training and testing set
from sklearn.cross_validation import train_test_split

# x_train and y_train represent the features and labels for the training set
# x_test and y_test represent the features and labels for the testing set
# test_size is saying what percentage of the data you want for testing .5 = 50%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)


# import the decision tree classifier
from sklearn import tree

# import the support vector machine classifier
from sklearn import svm

# import the k nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

# create the k nearest neighbors classifier
nearestNeighbors = KNeighborsClassifier()

# create the support vector machine classifier
supportVector = svm.SVC()

# create the classifier
decisionTree = tree.DecisionTreeClassifier()


# train the classifier
nearestNeighbors.fit(x_train, y_train)

# train the support vector machine
supportVector.fit(x_train, y_train)

# train the decisionTree
decisionTree.fit(x_train, y_train)

# store the predictions of the classifier
Npredictions = nearestNeighbors.predict(x_test)

# predict the results of the classifier
Dpredictions = decisionTree.predict(x_test)

# store the predictions form the support vector in this variable
Spredictions = supportVector.predict(x_test)

# print all of the labels which were predicted
### print predictions

# Check the accuruacy of the classifier on the testing data
from sklearn.metrics import accuracy_score

# print the accuracy score for the decision tree
print "The Decision tree classifier was %f%% accurate." % (accuracy_score(y_test, Dpredictions) * 100)

# Print the accuracy of the support vector machine
print "The Support Vectorm Machine Classifier was %f%% accurate." % (accuracy_score(y_test, Spredictions) * 100)

# print the accuracy of the k nearest neighbors algorithm
print "The K Nearest Neighbors Classifier was %f%% accurate." % (accuracy_score(y_test, Npredictions) * 100)
