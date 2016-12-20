
# import used for euclidean Distance formula
from scipy.spatial import distance


# K nearest neighbors uses euclidean distance formula to find the nereast neighbor
# this function will find the distance between points a and b
def euclideanDistance(a, b):
    return distance.euclidean(a, b)

# implement class for new classifier
# this will support two functions, fit to train the classfier and predict to test it
# in K nearest neighbors the k represents the number of neighbors you consider
class BareBonesKNN():

    # Method to train the algorithm, takes the training features and labels as input
    def fit(self, x_train, y_train):
        # Memorize the training data
        self.x_train = x_train
        self.y_train = y_train


    # Method implemented to make a prediction given a test point
    def predict(self, x_test):

        # since x_test is a 2d array we need to store the predictions in a 1 d array
        predictions = []

        # Go through the entire testing set and find the closest training points
        # to a test point, and then add that label to the predictions array
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    # Function which will find the closest training point to the test point
    # this method will loop over all the training point and keep track of the closest one
    def closest(self, row):

        # variable which keeps track of shortest distance
        # calculate the distance from the test point to the first training point
        shortestDistance = euclideanDistance(row, self.x_train[0])

        # variable which keeps track of the index to the shortest distance
        shortestIndex = 0

        # loop through the entire training set
        for i in range(1, len(self.x_train)):

            # Find the distance from the test point to all other training points
            distance = euclideanDistance(row, self.x_train[i])

            # if the new distance is smaller than our current shortest distance update shortest distance
            if distance < shortestDistance:
                shortestDistance = distance

                # since we have a new shortest distance match the index to the new current shortest index
                shortestIndex = i

        # return the label of the closest training example
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

# print the accuracy of the BareBonesKNN algorithm
print "The simple BareBonesKNN Classifier was %f%% accurate." % (accuracy_score(y_test, Npredictions) * 100)
