
from scipy.spatial import distance



def euclideanDistance(a, b):
    return distance.euclidean(a, b)

# implement class for new classifier
# this will support two functions, fit to train the classfier and predict to test it
# in K nearest neighbors the k represents the number of neighbors you consider
class BareBonesKNN():

    # Method to train the algorithm, takes the training features and labels as input
    def fit(self, x_train, y_train):
        # store the training elements for now
        self.x_train = x_train
        self.y_train = y_train


    # Method to test our algorithm
    def predict(self, x_test):
        # since x_test is a 2d array we need to store the predictions in a 1 d array
        predictions = []

        # for now just randomly choose a label from the testing data
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions


    def closest(self, row):
        shortestDistance = euclideanDistance(row, self.x_train[0])
        shortestIndex = 0

        for i in range(1, len(self.x_train)):
            distance = euclideanDistance(row, self.x_train[i])
            if distance < shortestDistance:
                shortestDistance = distance
                shortestIndex = i

        return self.y_train[shortestIndex]


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

# create the BareBonesKNN classifier
KNNClassifier = BareBonesKNN()

# train the classifier
KNNClassifier.fit(x_train, y_train)

# store the predictions of the classifier
Npredictions = KNNClassifier.predict(x_test)

# Check the accuruacy of the classifier on the testing data
from sklearn.metrics import accuracy_score

# print the accuracy of the k nearest neighbors algorithm
print "The K Nearest Neighbors Classifier was %f%% accurate." % (accuracy_score(y_test, Npredictions) * 100)
