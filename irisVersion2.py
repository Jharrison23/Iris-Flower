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

# create the classifier
decisionTree = tree.DecisionTreeClassifier()

# train the decisionTree
decisionTree.fit(x_train, y_train)
# predict the results of the classifier
predictions = decisionTree.predict(x_test)

# print all of the labels which were predicted
print predictions
