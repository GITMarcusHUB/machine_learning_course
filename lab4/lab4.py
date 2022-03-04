"""
Polynomial regression with scikit learn
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# creating a dictionary for data sets
dataset_paths = {"metrics": '/content/poly.npz'}

dataset = np.load(dataset_paths['metrics'])

X = dataset['X']
Y = dataset['Y']
X_test = dataset['X_test']
Y_test = dataset['Y_test']

print("X shape: \n", X.shape)
print("Y shape: \n", Y.shape)
print("X \n", X, "\n")
print("y \n", Y, "\n")
print("X_test shape \n", X_test.shape, "\n")
print("Y_test shape \n", Y_test.shape, "\n")
print("X_test \n", X_test, "\n")
print("Y_test \n", Y_test, "\n")

poly1 = Pipeline([('polynomial', PolynomialFeatures(degree=1)),
                  ('linear_regression', LinearRegression(fit_intercept=False))])
poly2 = Pipeline([('polynomial', PolynomialFeatures(degree=2)),
                  ('linear_regression', LinearRegression(fit_intercept=False))])
poly5 = Pipeline([('polynomial', PolynomialFeatures(degree=5)),
                  ('linear_regression', LinearRegression(fit_intercept=False))])

poly1 = poly1.fit(X, Y)
poly2 = poly2.fit(X, Y)
poly5 = poly5.fit(X, Y)

first_order = poly1.named_steps['linear_regression'].coef_
second_order = poly2.named_steps['linear_regression'].coef_
fifth_order = poly5.named_steps['linear_regression'].coef_


print("1st order polynomial regression \n", first_order)
print("2nd order polynomial regression \n", second_order)
print("5th order polynomial regression \n", fifth_order, "\n")

poly1_test = poly1.fit(X_test, Y_test)
poly2_test = poly2.fit(X_test, Y_test)
poly5_test = poly5.fit(X_test, Y_test)

first_order_test = poly1_test.named_steps['linear_regression'].coef_
second_order_test = poly2_test.named_steps['linear_regression'].coef_
fifth_order_test = poly5_test.named_steps['linear_regression'].coef_

print("1st order polynomial regression on test set\n", first_order_test)
print("2nd order polynomial regression on test set\n", second_order_test)
print("5th order polynomial regression on test set\n", fifth_order_test, "\n")

# TODO train and test MSE

y1 = first_order[0] + first_order[1] * X
y2 = second_order[0] + second_order[1] * X + second_order[2] * X**2
y5 = fifth_order[0] + fifth_order[1] * X + fifth_order[2]*X**2 + fifth_order[3]*X**3 + fifth_order[4]*X**4 + fifth_order[5]*X**5

y1_test = first_order_test[0] + first_order_test[1] * X_test
y2_test = second_order_test[0] + second_order_test[1] * X_test + second_order_test[2] * X_test**2
y5_test = fifth_order_test[0] + fifth_order_test[1] * X_test + fifth_order_test[2]*X_test**2 + fifth_order_test[3]*X_test**3 + fifth_order_test[4]*X_test**4 + fifth_order_test[5]*X_test**5

plt.scatter(X, Y)
plt.plot(X, y1, X, y2, X, y5)
plt.scatter(X_test,Y_test)
plt.plot(X_test, y1_test, X_test, y2_test, X_test, y5_test)
plt.show()

X = np.linspace(-5*math.pi, 5*math.pi, 500)
Y = np.expand_dims(np.sin(X),1)
degree = 3
dataset_size = np.random.randint(degree+1,100)

X_train = np.random.choice(X, dataset_size)
Y_train = np.expand_dims(np.sin(X_train),1)

w = np.random.rand(degree+1, 1)


phi = np.ones((1, X_train.shape[0]))
for k in range(1, degree+1):
    curr = np.power((np.squeeze(X_train)), k*np.ones_like((np.squeeze(X_train))))
    phi = np.concatenate((phi, np.expand_dims(curr,0)), 0)


phi_all = np.ones((1, X.shape[0]))
for k in range(1, degree+1):
    curr = np.power((np.squeeze(X)), k*np.ones_like((np.squeeze(X))))
    phi_all = np.concatenate((phi_all, np.expand_dims(curr,0)), 0)

epochs = 50
error_in = 0
error_out = 0
for i in range(epochs):
    pr = phi.transpose()@w-Y_train
    s = 0.01/X_train.shape[0]*(phi @ pr)
    w = w-s