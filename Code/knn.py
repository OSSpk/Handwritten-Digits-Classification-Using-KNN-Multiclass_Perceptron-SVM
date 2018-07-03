from mnist import MNIST
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

import time


# --------------------------------------------------------

# Global Variables

training_size = 3000
validation_size = 1000
testing_size = 1000

# --------------------------------------------------------

"""
Predicts the instances labels using top k neighbours
"""


def predict(neighbours, k):
    top_k = [Counter(x[:k]) for x in neighbours]
    predicted_labels = [x.most_common(1)[0][0] for x in top_k]

    return predicted_labels


# --------------------------------------------------------

"""
Finds the optimal value of k using cross validation.
The value of k with minimum error is the optimal one
"""


def find_k(neighbours, real_validation_labels, similarity_measure):
    k_values = []
    error_values = []

    real_validation_labels = list(real_validation_labels)

    """
    Its a convention to start from k = 1 to k = sqrt(N) where N is the size of training data
    """
    for k in range(math.ceil(math.sqrt(training_size))):
        k += 1

        predicted_labels = predict(neighbours, k)

        # check accuracy
        acc = accuracy_score(real_validation_labels, predicted_labels)

        k_values.append(k)
        error_values.append(1 - acc)

    if similarity_measure == 1:
        s = "Cosine Similarity"
    else:
        s = "Euclidean Distance"

    k = k_values[np.argmin(error_values)]



    """ Plotting the Validation Error Curve """

    plt.ylabel('Validation Error', fontsize=14)
    plt.xlabel('K', fontsize=14)
    plt.title("Validation Error Curve using %s" % s, fontsize=16, color='green')
    plt.plot(k_values, error_values, 'bo--')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(13, 7)

    plt.savefig("Validation Error Curve using %s.png" % s, dpi=300)
    plt.clf()

    """
    The value of K which gave minimum validation error is the optimal value of k
    """
    return k_values[np.argmin(error_values)]


# --------------------------------------------------------


def knn(train_images, test_images, train_labels, similarity_measure):
    if similarity_measure == 1:
        # compute cosine similarity
        v = [[np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y)) for y in train_images] for x in test_images]
        # v = cosine_similarity(test_images, train_images)
        r = True
    else:
        # compute euclidean distance
        v = [[np.sum((x - y) ** 2) for y in train_images] for x in test_images]
        r = False

    # append labels
    v = [[(x[i], train_labels[i]) for i in range(len(x))] for x in v]

    # sort in descending order
    [x.sort(key=lambda y: y[0], reverse=r) for x in v]

    # get all neighbours
    neighbours = [[n for similarity, n in x] for x in v]

    return neighbours


# --------------------------------------------------------

"""
Note: This is an experiment in which first the optimal value of K is determined using the
      cross validation technique and then its used to classify test images. This experiment
      is run twice. Once for Cosine Similarity and once for Euclidean Distance.
"""


def run_experiment(train_images, train_labels, test_images, test_labels, validation_images, validation_labels,
                   similarity_measure):
    """
    First finding the optimal value of K using validation images
    and then using it to classify test images
    """

    if similarity_measure == 1:
        s = "Cosine Similarity"
    else:
        s = "Euclidean Distance"

    print("------------------------------------------")
    print("Running Experiment using %s" % s)
    print("------------------------------------------")

    print("Finding Optimal Value of K ...")
    neighbours_labels = knn(train_images, validation_images, train_labels, similarity_measure)
    k = find_k(neighbours_labels, validation_labels, similarity_measure)

    print("Optimal Value of K using Cross Validation is: %d" % k)
    print("Classifying Test Images ...")

    start_time = time.clock()

    neighbours_labels = knn(train_images, test_images, train_labels, similarity_measure)
    predicted_labels = predict(neighbours_labels, k)

    print("Prediction Time: %.2f seconds" % (time.clock() - start_time))

    print("Test Images Classified!")
    accuracy = accuracy_score(test_labels, predicted_labels) * 100

    print("KNN with k = %d" % k)
    print("Accuracy: %f" % accuracy, "%")
    print("---------------------\n")


# --------------------------------------------------------


def main():
    # load data
    data = MNIST('samples')

    train_images, train_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    validation_images = np.array(train_images[training_size:training_size + validation_size])
    validation_labels = np.array(train_labels[training_size:training_size + validation_size])

    train_images = np.array(train_images[:training_size])
    train_labels = np.array(train_labels[:training_size])

    test_images = np.array(test_images[:testing_size])
    test_labels = np.array(test_labels[:testing_size])

    """Rescaling Data"""
    train_images = train_images/255
    test_images = test_images/255
    validation_images = validation_images/255

    """ run knn with cosine similarity as similarity measure """
    run_experiment(train_images, train_labels, test_images, test_labels, validation_images, validation_labels, 1)

    """ run knn with euclidean distance as similarity measure """
    run_experiment(train_images, train_labels, test_images, test_labels, validation_images, validation_labels, 2)


# --------------------------------------------------------


# get things rolling
main()
