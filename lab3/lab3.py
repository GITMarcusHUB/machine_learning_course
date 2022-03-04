"""
Márk Pető
Logistic regression vs. linear regression
"""
import numpy as np
import matplotlib.pyplot as plt


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

# creating a dictionary for data sets
dataset_paths = {"metrics": '/content/metrics.npz'}

# Data & random weights
dataset = np.load(dataset_paths['metrics'])

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
visualize(weights, X, Y*2-1)


# Linear regression
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
weights = np.dot(
                np.dot(
                    np.linalg.inv(
                        np.dot(X.T, X)),
                X.T),
             Y)

MSE = calculate_mse(weights, X, Y)
random_weights = np.random.rand(*weights.shape) * (np.subtract(np.max(weights),np.min(weights)))
random_mse = calculate_mse(random_weights, X, Y)
val_mse = calculate_mse(weights, X_test, Y_test)

print("Validation MSE: ", val_mse)

visualize(weights, X, Y*2-1)

# Logistic regression
batch_size = 32
alpha = 0.01
iterations = 2500
print_freq = 500
epsilon = 1e-5
epochs = 50


def sigmoid(x):
    return scipy.special.expit(x)


def gradient_descent(X, Y, weights, alpha, iterations):
    """
    Calculates gradient descent algorithm, return the new weights after the given number of iterations
    """
    score = np.rint(sigmoid(np.dot(X, weights)))
    for i in range(iterations):
        weights = weights - alpha * np.dot(X.T, (score - Y))
        if i % print_freq == 0:
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
    error = -np.average((((Y).T * np.log(prediction + epsilon)) + ((1 - Y).T * np.log(1 - prediction + epsilon))))
    return error


weights = np.random.randn(3, 1)
# weights = gradient_descent(X, Y, weights, alpha, iterations)
for it in range(1000):
    weights = sgd(batch_size, alpha, X, Y, weights)
loss = calc_loss(X, Y, weights)
print('loss:', loss)
visualize(weights, X, Y * 2 - 1)

