import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def classify_data(X, Y, classification_method, hyperparam_ranges, pre_processing):
    """Robust function for testing classification methods

   Args:
       X: List of input images

       Y: List of ground truth classifications

       classification_method: Given classification method you want to test.
        The function's parameters must adhere to the following convention:
        classification_method(trainX, trainY, testX, testY, [hyperparamerts])
        and returns the predictedY and loss value

       hyperparam_ranges: List of ranges of each classification hyperparameter.
        For example, for hyperparameters a and b:
        [[a1,...,a5], [b1,...,b3]]

       pre_processing: List of preprocessing algorithms along with any required parameters.
        For example, for preprocessing algorithm algo1 with parameters a and b:
        [[algo1, [a, b]],...,other_algorithms]
        As such, preprocessing algorithms must adhere to the following convention:
        preprocessing_algorithm(input_image, [list_of_parameters])
        and returns the preprocessed image
   """

    # Pre-process Data
    for i in range(X):
        for pre_process in pre_processing:
            X[i] = pre_process[0](X[i], pre_process[1])

    # Randomize & Split Data
    testX, testY, trainX, trainY, valX, valY = randomize_data(X, Y)

    # Optimize Hyperparams
    best_loss = 1234567890
    best_param = None
    hyperparams = itertools.product(*hyperparam_ranges)
    for hyperparam in hyperparams:
        _, loss = classification_method(trainX, trainY, valX, valY, hyperparam)
        if loss < best_loss:
            best_loss = loss
            best_param = hyperparam

    # Classify Data
    predictedY, _ = classification_method(trainX, trainY, testX, testY, best_param)

    # Print Accuracy & Generate Confusion Matrix
    report_results(predictedY, testY)


def randomize_data(X, Y):

    # Randomize Data
    shuffler = np.random.permutation(len(X))
    shuffledX = X[shuffler]
    shuffledY = Y[shuffler]

    # Initialize test data
    testX = shuffledX[0:int(len(shuffledX) / 5)]
    testY = shuffledY[0:int(len(shuffledY) / 5)]

    # Initialize validation data
    valX = shuffledX[int(len(shuffledX) / 5):int(len(shuffledX) * 2 / 5)]
    valY = shuffledY[0:int(len(shuffledY) / 5):int(len(shuffledY) * 2 / 5)]

    # Initialize training data
    trainX = shuffledX[int(len(shuffledX) * 2 / 5): -1]
    trainY = shuffledY[int(len(shuffledY) * 2 / 5): -1]

    return testX, testY, trainX, trainY, valX, valY


def report_results(predictedY, testY):
    num_classes = np.max(testY)
    incorrect_indices = []
    correct_indices = []
    confusion_data = np.zeros([num_classes, num_classes])

    for i, predicted_class in enumerate(predictedY):
        if predicted_class != np.argmax(testY[i]):
            incorrect_indices.append(i)
        else:
            correct_indices.append(i)
        confusion_data[predicted_class][np.argmax(testY[i])] += 1
    precision_data = np.zeros(num_classes)
    recall_data = np.zeros(num_classes)

    for x, arr in enumerate(confusion_data):
        sum = 0
        diagonal = 0
        for y, val in enumerate(arr):
            sum += val
            if x == y:
                diagonal = val
        precision_data[x] = diagonal / sum

    for y in range(num_classes):
        sum = 0
        diagonal = 0
        for x in range(num_classes):
            sum += confusion_data[x][y]
            if x == y:
                diagonal = confusion_data[x][y]
        recall_data[y] = diagonal / sum

    heat_map = sns.heatmap(confusion_data, cmap='Blues', annot=True, linewidths=0, fmt='g')
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.title("Confusion Matrix")
    plt.show()