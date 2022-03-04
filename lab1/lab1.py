"""
Pető Márk
T43V16
Lab 1 of Machine Learning course
2020.02.14.

- mean squared error
- linear regression
- bias
- PLA algorithm
- Pocket algorithm
"""
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse(weights, data, labels):
    """
    Calculates means squared error (L2 loss) from model weights & input
    :param weights: model weights
    :param data: input data (x)
    :param label: target label (y)
    return: mse value (np.float64)
    """
    mse = np.average(
                np.square(
                    np.subtract(
                        np.dot(data, weights), labels)))
    return mse


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


def task1(X, Y, weights, X_test, Y_test, verbose):
    # Task 1 - plain linear regression, I assume the parameters correct, if there is a sufficiently
    # large difference between the random error and the error calculated from the weights (> threshold=1e2)

    MSE = calculate_mse(weights, X, Y)
    random_weights = np.random.rand(*weights.shape) * (np.subtract(np.max(weights), np.min(weights)))
    random_mse = calculate_mse(random_weights, X, Y)
    val_mse = calculate_mse(weights, X_test, Y_test)

    if verbose:
        print("Task 1")
        print("Shape of trainng set of x:", X.shape)
        print("Shape of test set of x:", X_test.shape)
        print("Shape of trainng set of y:", Y.shape)
        print("Shape of test set of y:", Y_test.shape)

        print("Content of trainng set of x:", X)
        print("Content of test set of x:", X_test)
        print("Content of trainng set of y:", Y)
        print("Content of test set of y:", Y_test, "\n")

        print("Weights: ", weights)

        print("Training loss (MSE)")
        print("Training MSE: ", MSE)
        print("Random weights: ", random_weights)
        print("MSE on random weights: ", random_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(MSE, random_mse)))

        print("Validation loss (MSE)")
        print("Validation MSE: ", val_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(val_mse, random_mse)))
        print("Result of the prediction (y = Wx):")
        print(np.dot(X, weights))


def task2(X, Y, weights, X_test, Y_test, verbose):
    # Task 2 - linear regression with bias, I assume the parameters correct, if there is a sufficiently
    # large difference between the random error and the error calculated from the weights (> threshold=1e2)

    MSE = calculate_mse(weights, X, Y)
    random_weights = np.random.rand(*weights.shape) * (np.subtract(np.max(weights), np.min(weights)))
    random_mse = calculate_mse(random_weights, X, Y)
    val_mse = calculate_mse(weights, X_test, Y_test)

    if verbose:
        print("Task 2")
        print("Shape of trainng set of x:", X.shape)
        print("Shape of test set of x:", X_test.shape)
        print("Shape of trainng set of y:", Y.shape)
        print("Shape of test set of y:", Y_test.shape)

        print("Content of trainng set of x:", X)
        print("Content of test set of x:", X_test)
        print("Content of trainng set of y:", Y)
        print("Content of test set of y:", Y_test, "\n")

        print("Weights: ", weights)

        print("Training loss (MSE)")
        print("Training MSE: ", MSE)
        print("Random weights: ", random_weights)
        print("MSE on random weights: ", random_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(MSE, random_mse)))

        print("Validation loss (MSE)")
        print("Validation MSE: ", val_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(val_mse, random_mse)))
        print("Result of the prediction (y = Wx):")
        print(np.dot(X, weights))


def task3(X, Y, weights, X_test, Y_test, verbose):
    # Task 3 - Note that both the training & validation loss is increasing usign 'affine2.npz' dataset
    # This could be because 'affine1.npz' is an artificial dataset with the affine transformation
    # that the bias can model exceptionally well (addition). Other affine transformations (e.g. rotation) can cause more
    # imprecise predictions.

    MSE = calculate_mse(weights, X, Y)
    random_weights = np.random.rand(*weights.shape) * (np.subtract(np.max(weights), np.min(weights)))
    random_mse = calculate_mse(random_weights, X, Y)
    val_mse = calculate_mse(weights, X_test, Y_test)

    if verbose:
        print("Task 3")
        print("Shape of trainng set of x:", X.shape)
        print("Shape of test set of x:", X_test.shape)
        print("Shape of trainng set of y:", Y.shape)
        print("Shape of test set of y:", Y_test.shape)

        print("Content of trainng set of x:", X)
        print("Content of test set of x:", X_test)
        print("Content of trainng set of y:", Y)
        print("Content of test set of y:", Y_test, "\n")

        print("Weights: ", weights)

        print("Training loss (MSE)")
        print("Training MSE: ", MSE)
        print("Random weights: ", random_weights)
        print("MSE on random weights: ", random_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(MSE, random_mse)))

        print("Validation loss (MSE)")
        print("Validation MSE: ", val_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(val_mse, random_mse)))
        print("Result of the prediction (y = Wx):")
        print(np.dot(X, weights))


def task4a(X, Y, weights, X_test, Y_test, verbose):

    MSE = calculate_mse(weights, X, Y)
    random_weights = np.random.rand(*weights.shape) * (np.subtract(np.max(weights), np.min(weights)))
    random_mse = calculate_mse(random_weights, X, Y)
    val_mse = calculate_mse(weights, X_test, Y_test)

    if verbose:
        print("Task 4 - pla")
        print("Shape of trainng set of x:", X.shape)
        print("Shape of test set of x:", X_test.shape)
        print("Shape of trainng set of y:", Y.shape)
        print("Shape of test set of y:", Y_test.shape)

        print("Content of trainng set of x:", X)
        print("Content of test set of x:", X_test)
        print("Content of trainng set of y:", Y)
        print("Content of test set of y:", Y_test, "\n")

        print("Weights: ", weights)

        print("Training loss (MSE)")
        print("Training MSE: ", MSE)
        print("Random weights: ", random_weights)
        print("MSE on random weights: ", random_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(MSE, random_mse)))

        print("Validation loss (MSE)")
        print("Validation MSE: ", val_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(val_mse, random_mse)))
        print("Result of the prediction (y = Xw):")
        print(np.dot(X, weights))
        visualize(weights, X, Y)


def task4b(X, Y, weights, X_test, Y_test, verbose):
    # Task 4 - pocket, linear separator, two positive samples are reamined in the 'negative'

    MSE = calculate_mse(weights, X, Y)
    random_weights = np.random.rand(*weights.shape) * (np.subtract(np.max(weights), np.min(weights)))
    random_mse = calculate_mse(random_weights, X, Y)
    val_mse = calculate_mse(weights, X_test, Y_test)

    if verbose:
        print("Task 4 - pocket")
        print("Shape of trainng set of x:", X.shape)
        print("Shape of test set of x:", X_test.shape)
        print("Shape of trainng set of y:", Y.shape)
        print("Shape of test set of y:", Y_test.shape)

        print("Content of trainng set of x:", X)
        print("Content of test set of x:", X_test)
        print("Content of trainng set of y:", Y)
        print("Content of test set of y:", Y_test, "\n")

        print("Weights: ", weights)

        print("Training loss (MSE)")
        print("Training MSE: ", MSE)
        print("Random weights: ", random_weights)
        print("MSE on random weights: ", random_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(MSE, random_mse)))

        print("Validation loss (MSE)")
        print("Validation MSE: ", val_mse)
        print("Average difference between MSE of linear regression weights and randomly generated weights: ",
              np.abs(np.subtract(val_mse, random_mse)))
        print("Result of the prediction (y = Xw):")
        print(np.dot(X, weights))
        visualize(weights, X, Y)


def main():
    # creating a dictionary for data sets
    dataset_paths = {"linear": '/content/linear.npz',
                     "affine1": '/content/affine1.npz',
                     "affine2": '/content/affine2.npz',
                     "pla": '/content/pla.npz',
                     "pocket": '/content/pocket.npz'}
    # controls verbosity in the notebook
    verbose = True

    dataset = np.load(dataset_paths["linear"])
    X = dataset['X']
    Y = dataset['Y']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    weights = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X.T, X)),
            X.T),
        Y)

    task1(X, Y, weights, X_test, Y_test, verbose)

    dataset = np.load(dataset_paths["affine1"])
    X = dataset['X']
    Y = dataset['Y']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    weights = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X.T, X)),
            X.T),
        Y)

    task2(X, Y, weights, X_test, Y_test, verbose)

    dataset = np.load(dataset_paths["affine2"])
    X = dataset['X']
    Y = dataset['Y']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    weights = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X.T, X)),
            X.T),
        Y)

    task3(X, Y, weights, X_test, Y_test, verbose)

    task4a(X, Y, weights, X_test, Y_test, verbose)
    dataset = np.load(dataset_paths["pocket"])
    X = dataset['X']
    Y = dataset['Y']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)

    weights = np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X.T, X)),
            X.T),
        Y)

    task4b(X, Y, weights, X_test, Y_test, verbose)


if __name__ == "__main__":
    main()