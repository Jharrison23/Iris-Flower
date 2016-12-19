# Machine Learning Classification algorithm to identify the type of flower
# based off of the features Sepal Length and Width, and Petal Length and Width
# Written by James Harrison following along with a google researcher tutorial

# Data set obtained from the URL https://en.wikipedia.org/wiki/Iris_flower_data_set

# we will import the data set to train our classifier and then use that classifier
# to predict what species of flower we have if we get a new flower we havent seen before


# Iris is included in the sample datasets for scikit learn so we can import it
from sklearn.datasets import load_iris

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
