import warnings

import numpy as np
from mnist import MNIST
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

import time

warnings.filterwarnings("ignore")

# --------------------------------------------------------

# Global Variables

training_size = 5000
testing_size = 10000

alpha = 0.01
iterations = 2000  # epochs for batch mode gradient descent

gradient_curve = True
epoch_curve = True

"""Since we are not sure that the dataset is linearly separable or not, therefore to avoid infinite looping in weights training
we iterate over the training set epoch times for tuning weights"""
epochs = 15  # epochs for perceptron rule (currently stochastic mode is used)

labels = 10

# --------------------------------------------------------

"""Predicts the labels by choosing the label of the classifier with highest confidence(probability)"""


def predict(all_weights, test_images, learning_algo):
    test_images = np.hstack((np.ones((testing_size, 1)), test_images))

    predicted_labels = np.dot(all_weights, test_images.T)

    # sigmoid activation function
    if learning_algo == 1:
        predicted_labels = sigmoid(predicted_labels)

    # signum activation function
    elif learning_algo == 2:
        predicted_labels = signum(predicted_labels)

    predicted_labels = np.argmax(predicted_labels, axis=0)

    return predicted_labels.T


# --------------------------------------------------------

"""Activation Functions"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def signum(x):
    x[x > 0] = 1
    x[x <= 0] = -1

    return x


# --------------------------------------------------------
def learn_using_perceptron_rule_with_signum_function(train_images, train_labels, weights):
    epochs_values = []
    error_values = []

    for k in range(epochs):
        missclassified = 0

        for t, l in zip(train_images, train_labels):
            h = np.dot(t, weights)

            h = signum(h)

            if h[0] != l[0]:
                missclassified += 1

            gradient = t * (h - l)

            # reshape gradient
            gradient = gradient.reshape(gradient.shape[0], 1)

            weights = weights - (gradient * alpha)

        error_values.append(missclassified / training_size)
        epochs_values.append(k)

    global epoch_curve

    if epoch_curve:
        """"Plotting Epochs vs Training Error"""

        plt.ylabel('Training Error', fontsize=14)
        plt.xlabel('Epoch', fontsize=14)
        plt.title("Epochs vs Training Error for Perceptron Rule with Signum Func.", fontsize=16, color='green')
        plt.plot(epochs_values, error_values)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(13, 7)

        plt.savefig("Epochs vs Training Error for Perceptron Rule with Signum Func..png", dpi=300)
        plt.clf()

        epoch_curve = False

    return weights


def learn_using_gradient_descent_with_sigmoid_function(train_images, train_labels, weights):
    iters = []
    error_values = []

    for i in range(iterations):
        h = np.dot(train_images, weights)

        h = sigmoid(h)

        error_value = (np.dot(-1 * train_labels.T, np.log(h)) - np.dot((1 - train_labels).T,
                                                                       np.log(1 - h))) / training_size

        gradient = np.dot(train_images.T, h - train_labels) / training_size

        weights = weights - (gradient * alpha)

        iters.append(i)
        error_values.append(error_value[0, 0])

    global gradient_curve

    if gradient_curve:
        """"Plotting Iterations vs Training Error"""

        plt.ylabel('Training Error', fontsize=14)
        plt.xlabel('Iteration', fontsize=14)
        plt.title("Iterations vs Training Error for Gradient Descent with Sigmoid Func.", fontsize=16, color='green')
        plt.plot(iters, error_values)
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(13, 7)

        plt.savefig("Iterations vs Training Error for Gradient Descent with Sigmoid Func.png", dpi=300)
        plt.clf()

        gradient_curve = False

    return weights


# --------------------------------------------------------

"""Find optimal weights for each logistic binary classifier"""


def train(train_images, train_labels, learning_algo):
    # add 1's as x0
    train_images = np.hstack((np.ones((training_size, 1)), train_images))

    # add w0 as 0 initially
    all_weights = np.zeros((labels, train_images.shape[1]))

    train_labels = train_labels.reshape((training_size, 1))

    train_labels_copy = np.copy(train_labels)

    for j in range(labels):

        print("Training Classifier: ", j+1)

        train_labels = np.copy(train_labels_copy)

        # initialize all weights to zero
        weights = np.zeros((train_images.shape[1], 1))

        if learning_algo == 1:
            for k in range(training_size):
                if train_labels[k, 0] == j:
                    train_labels[k, 0] = 1
                else:
                    train_labels[k, 0] = 0

            weights = learn_using_gradient_descent_with_sigmoid_function(train_images, train_labels, weights)

        elif learning_algo == 2:
            for k in range(training_size):
                if train_labels[k, 0] == j:
                    train_labels[k, 0] = 1
                else:
                    train_labels[k, 0] = -1

            weights = learn_using_perceptron_rule_with_signum_function(train_images, train_labels, weights)

        all_weights[j, :] = weights.T

    return all_weights


# --------------------------------------------------------

def run_experiment(train_images, train_labels, test_images, test_labels, learning_algo):
    if learning_algo == 1:
        s = "Gradient Descent with Sigmoid Activation Function"
    elif learning_algo == 2:
        s = "Perceptron Learning Rule for Thresholded Unit"

    print("------------------------------------------------------------------------------------")
    print("Running Experiment using %s" % s)
    print("------------------------------------------------------------------------------------")

    print("Training ...")
    start_time = time.clock()
    all_weights = train(train_images, train_labels, learning_algo)
    print("Training Time: %.2f seconds" % (time.clock() - start_time))
    print("Weights Learned!")

    print("Classifying Test Images ...")
    start_time = time.clock()
    predicted_labels = predict(all_weights, test_images, learning_algo)
    print("Prediction Time: %.2f seconds" % (time.clock() - start_time))

    print("Test Images Classified!")

    accuracy = accuracy_score(test_labels, predicted_labels) * 100

    print("Accuracy: %f" % accuracy, "%")
    print("---------------------\n")


# --------------------------------------------------------

def calculate_hog_features(images):
    list_hog_fd = [hog(t.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                       visualise=False) for t in images]

    return np.array(list_hog_fd, dtype=np.float64)


def run_svm_experiment(train_images, train_labels, test_images, test_labels):
    s = "SVM with HOG based feature descriptor"

    print("------------------------------------------------------------------------------------")
    print("Running Experiment using %s" % s)
    print("------------------------------------------------------------------------------------")

    print("Calculating HOG based Feature Descriptor")
    start_time = time.clock()
    train_hog_features = calculate_hog_features(train_images)
    print("Feature Descriptor calculated !")

    print("Training ...")
    clf = LinearSVC()
    clf.fit(train_hog_features, train_labels)
    print("Training Time: %.2f seconds" % (time.clock() - start_time))
    print("Training Done!")

    print("Classifying Test Images ...")
    start_time = time.clock()
    test_hog_features = calculate_hog_features(test_images)
    predicted_labels = clf.predict(test_hog_features)
    print("Prediction Time: %.2f seconds" % (time.clock() - start_time))

    print("Test Images Classified!")
    accuracy = accuracy_score(test_labels, predicted_labels) * 100

    print("Accuracy: %f" % accuracy, "%")
    print("---------------------\n")


# --------------------------------------------------------

def main():
    # load data
    data = MNIST('samples')

    train_images, train_labels = data.load_training()
    test_images, test_labels = data.load_testing()

    train_images = np.array(train_images[:training_size])
    train_labels = np.array(train_labels[:training_size], dtype=np.int32)

    test_images = np.array(test_images[:testing_size])
    test_labels = np.array(test_labels[:testing_size], dtype=np.int32)

    """Rescaling Data"""
    train_images = train_images / 255
    test_images = test_images / 255

    """ run one vs all with learning a perceptron with sigmoid activation function using gradient descent """
    run_experiment(train_images, train_labels, test_images, test_labels, 1)

    """ run one vs all with learning a perceptron with perceptron learning rule """
    run_experiment(train_images, train_labels, test_images, test_labels, 2)

    """ run svm using HOG as feature descriptor """
    run_svm_experiment(train_images, train_labels, test_images, test_labels)


# --------------------------------------------------------


# get things rolling
main()
