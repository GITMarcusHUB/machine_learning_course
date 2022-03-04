"""
Márk Pető
polynomial regression
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi

np.random.seed(42)

# 1 - plot learning curve  of polynomial regression , when approximating sine function

# X values
grid_points = np.linspace(-5*pi, 5*pi, num=500)
# test dataset, Y_test
test_dataset = np.sin(grid_points)

degree = 1
# dataset size
k = np.random.randn(degree+1, 100)
# training dataset


def poly_val(theta, x):
    val = theta[0]
    for i in range(1, len(theta)):
        val += theta[i]*x**i
    return val


def polynomial_regressions(degree=2, verbose=False, plotting=True, weight_decay=False, lambda_=0.1):
    k = np.random.randint(degree + 1, 100)
    training_grid = 10 * pi * np.random.random_sample(k) - 5 * pi
    training_dataset = np.sin(training_grid)
    experiment_number = 50
    ITERATION_NUM = 10
    error = 0
    learning_rate = 0.001

    if verbose:
        print(f'training grid shape: {training_grid.shape}')
        print(f'dataset_shape:{training_dataset.shape}')

    N = 100 - degree + 2
    error_ins = []
    error_outs = []
    for k in range(degree + 1, 100):
        # k = np.random.randint(degree+1, 100)
        training_grid = 10 * pi * np.random.random_sample(k) - 5 * pi
        training_dataset = np.sin(training_grid)
        theta = np.zeros(degree + 1)
        for it in range(ITERATION_NUM):
            hypothesis = poly_val(theta, training_grid)
            if weight_decay:
                theta_norm = theta - np.min(theta)
                theta_norm = np.divide(theta_norm, np.max(theta_norm))
                print(theta_norm)
                theta += -learning_rate * (1 / training_grid.shape[0]) * np.dot(
                    (hypothesis - training_dataset + lambda_ * theta_norm), training_grid)
            else:
                theta += -learning_rate * (1 / training_grid.shape[0]) * np.dot(hypothesis - training_dataset,
                                                                                training_grid)
        error_in = np.sum(np.square(hypothesis - training_dataset)) / training_dataset.shape[0]
        test_hypothesis = poly_val(theta, grid_points)
        error_out = np.sum(np.square(test_hypothesis - test_dataset)) / test_dataset.shape[0]

        error_ins.append(error_in)

        error_outs.append(error_out)
        if k % 25 == 0 and verbose:
            print(f'Experiment{k} hypothesis shape: {hypothesis.shape}')
            print(f'Experiment{k} hypothesis:\n {hypothesis}')
            # print(f'In-sample error: {error_in}')
            # print(f'Out-of-sample error: {error_out}')
    if verbose:
        print(f'In sample errors: {error_ins}')
        print(f'Out of sample errors: {error_outs}')

        print(f'Average in sample errors: {np.mean(np.asarray(error_ins))}')
        print(f'Average out sample errors: {np.mean(np.asarray(error_outs))}')
    if plotting:
        plt.figure(1)
        plt.title(f"Learning_curve with degree {degree}")
        plt.xlabel("dataset sizes")
        plt.ylabel("L2 error rates")
        plt.plot(range(degree + 1, 100), error_ins, c="r", linewidth=2, label='In sample error')
        plt.plot(range(degree + 1, 100), error_outs, c="b", linewidth=2, label='Out sample error')
        plt.legend()
        plt.show()

polynomial_regressions(degree=1)
polynomial_regressions(degree=2)
polynomial_regressions(degree=5)
polynomial_regressions(degree=8)
# If we have only 20 points in the dataset, I would choose degree 2, because with degrees 5 and 8, there are spikes on the graphs after 20 points

# Unfortunately, I couldn't do this
polynomial_regressions(degree=1, weight_decay=True)
# polynomial_regressions(degree=2, weight_decay=True)
# polynomial_regressions(degree=5, weight_decay=True)
# polynomial_regressions(degree=8, weight_decay=True)

