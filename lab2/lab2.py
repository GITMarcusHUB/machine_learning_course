"""
Márk Pető
TODO upload data content

- gradient descent
- stochastic gradient descent
- PLA algorithm
- Pocket algorithm
- visualization
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import scipy.special


def visualize(w, X, Y):
    """
    Visualises the dataset together with the separator

    Note: this is only for debugging, it is not the figure expected in the
    exercise.

    # Parameters
    w (array-like): the model weights (a vector with 3 elements)
    X (array-like): an Nx2 matrix of input elements
    Y (array-like): an Nx1 matrix of labels
    """
    def separator_points(w):
        l = -w[2] / (w[0]**2 + w[1]**2)
        x0 = np.array([l * w[0], l * w[1]])
        v = np.array([-w[1], w[0]])
        v /= np.linalg.norm(v, ord = 2)
        return np.stack([x0 + 5 * v, x0 - 5 * v])
    w = w.reshape((-1,))
    fig, ax = plt.subplots()
    positive_samples = X[Y[:, 0] == 1, :]
    negative_samples = X[Y[:, 0] == -1, :]
    ax.plot(positive_samples[:, 0], positive_samples[:, 1], 'x',
            label = '$+1$')
    ax.plot(negative_samples[:, 0], negative_samples[:, 1], 'o',
            label = '$-1$')
    if np.linalg.norm(w) < 1e-10 or np.linalg.norm(w[:2]) < 1e-10:
        print('Warning: the norm of the weight vector is to small to plot.')
    else:
        s = separator_points(w)
        ax.plot(s[:, 0], s[:, 1], '-', label = 'separator')
    ax.legend()
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.grid(True)
    fig.tight_layout()
    plt.show()


def sigmoid(x):
    return scipy.special.expit(x)


def gradient_descent(X, Y, weights, alpha, iterations):
    """
    Calculates gradient descent algorithm, return the new weights after the given number of iterations
    """
    score = np.rint(sigmoid(np.dot(X, weights)))
    for i in range(iterations):
        weights = weights - alpha*np.dot(X.T, (score - Y))
        if i%print_freq == 0:
            visualize(weights, X, Y)

    return weights


def sgd(batch_size, alpha, X, Y, weights):
    """
    Calculates stochastic gradient descent algorithm, return the new weights
    """
    indices = np.arange(0, X.shape[0] - 1)
    np.random.shuffle(indices)

    # cut off remaining samples
    # indices = indices[0:(X.shape[0] - X.shape[0]%batch_size)]
    shards = np.split(indices, len(indices) / batch_size)
    sum_loss = 0
    for i in range(len(shards)):
        prediction = sigmoid(np.dot(X[shards[i]], weights))
        sum_loss += np.dot(X[shards[i]].T, Y[shards[i]] * (prediction - 1) + (1 - Y[shards[i]]) * prediction)
    sum_loss = sum_loss / batch_size

    weights = weights - alpha * sum_loss
    return weights


def calc_loss(X, Y, weights):
    prediction = np.rint(sigmoid(np.dot(X, weights)))
    error = -np.average((((Y).T * np.log(prediction + epsilon)) + ((1-Y).T * np.log(1-prediction + epsilon))))
    return error


def main():
    # creating a dictionary for data sets
    dataset_paths = {"pla": '/content/pla.npz',
                     "pocket":'/content/pocket.npz',
                     "yesno":'/content/yesno.npz'}
    # controls verbosity in the notebook
    verbose = True
    batch_size = 32
    alpha = 0.01
    iterations = 2500
    print_freq = 500
    epsilon = 1e-5
    epochs = 50

    # Task 1 - PLA
    dataset = np.load(dataset_paths["pla"])
    X = dataset['X']
    Y = dataset['Y']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    print(X.shape)
    print(Y.shape)
    print(X[0])
    print(Y[0])

    weights = np.random.randn(3, 1)
    init_loss = calc_loss(X, Y, weights)
    print('init_loss: ', init_loss)
    weights = gradient_descent(X, Y, weights, alpha, iterations)
    loss = calc_loss(X, Y, weights)
    print('loss:', loss)
    visualize(weights, X, Y)

    # Task 1 - Pocket
    dataset = np.load(dataset_paths["pocket"])
    X = dataset['X']
    Y = dataset['Y']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    print(X.shape)
    print(Y.shape)
    print(X[0])
    print(Y[0])

    weights = np.random.randn(3, 1)
    init_loss = calc_loss(X, Y, weights)
    print('init_loss: ', init_loss)
    weights = gradient_descent(X, Y, weights, alpha, iterations)
    loss = calc_loss(X, Y, weights)
    print('loss:', loss)
    visualize(weights, X, Y)

    # Task 2 - Speech recognition
    speech_dataset = np.load(dataset_paths["yesno"])
    X = speech_dataset['X_train']
    Y = speech_dataset['Y_train']
    X_test = speech_dataset['X_test']
    Y_test = speech_dataset['Y_test']

    # apply FFT + keeping only magnitudes
    X = np.abs(scipy.fftpack.rfft(X))
    X_test = np.abs(scipy.fftpack.rfft(X_test))

    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    Y = np.expand_dims(Y, 1)
    Y_test = np.expand_dims(Y_test, 1)

    print('X shape: ', X.shape)
    print('Y shape: ', Y.shape)
    print('X_test shape:', X_test.shape)
    print('Y_test shape:', Y_test.shape)

    weights = np.random.randn(1001, 1)

    init_loss = calc_loss(X, Y, weights)
    print("init_loss: ", init_loss)

    # train_split = 0.7
    # validation_split = 0.3
    # train_size = int(X.shape[1]*train_split)
    # print("Training set length is: ", train_size)
    # print("Validation set length is: ", int(X.shape[1]*validation_split))


    # train_set = X[:, indices[:train_size]]
    # validation_set = X[:, indices[train_size:]]
    # test set is distinct already

    # training
    for i in range(epochs):
        weights = sgd(batch_size, alpha, X, Y, weights)
        mse = np.mean((sigmoid(np.dot(X_test, weights)) - Y_test)**2)

        if i % 2 == 0:
            print("mse: ", mse)

    batch_size = 16
    learning_rate = 1e-4

    # training
    for i in range(epochs):
        weights = sgd(batch_size, alpha, X, Y, weights)
        mse = np.mean((sigmoid(np.dot(X_test, weights)) - Y_test)**2)

        if i % 2 == 0:
            print("mse: ", mse)

if __name__ == "__main__":
    main()