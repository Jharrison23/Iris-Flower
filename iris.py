# Machine Learning Classification algorithm to identify the type of flower
# based off of the features Sepal Length and Width, and Petal Length and Width
# Written by James Harrison following along with a google researcher tutorial

# Data set obtained from the URL https://en.wikipedia.org/wiki/Iris_flower_data_set

# we will import the data set to train our classifier and then use that classifier
# to predict what species of flower we have if we get a new flower we havent seen before


# Iris is included in the sample datasets for scikit learn so we can import it
from sklearn.datasets import load_iris

# import decision tree classifier to be used for Classification
from sklearn import tree

import numpy as np
# load and return the iris dataset
# this data set includes the data from the link as well as meta data, which tells you the
# names of the features and the names of the flowers
iris = load_iris()

# print the list of features in the dataset
print iris.feature_names

# print the labels for this dataset
print iris.target_names

# Data variable stores the features and examples themselves, with an index representing
# their position in the table
# this will print the measurements which correspond with the features_names
print iris.data[0]

# The target variable contains the labels with the index also representing the table position
# this will display the first flower which has the features of data[0]
# it will return an index which corresponds to the array in target_names
print iris.target[0]

# Loop through all of the dataset and print the entire table including the labels and features
# added 1 to the i so that i said example 1 instead of example 0
for i in range(len(iris.data)):
    print "Example %d : label %s, features %s" % (i + 1, iris.target[i], iris.data[i])


# Train classifier
# first split data into testing set and training data
# Store some elements from the dataset into the test set
test_idx = [0, 50, 100]

# remove the entries which are stored in the test set from the original data and target variables
# this set of variables represent the training set without the entries from the testing set
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

# These variables represent the testing set
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
