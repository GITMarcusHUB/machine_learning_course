"""
MLP
Convolutional neural network
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.activations import relu, softmax


def get_dataset():
    # Change this to download somewhere else
    path = '/content/mnist.npz'
    (X, Y), (X_test, Y_test) = mnist.load_data(path)
    return X, Y, X_test, Y_test


def preprocess_data(X, Y):
    # TODO: normalise input values to between 0 and 1
    X = X.astype(np.float32)
    X = X/255
    # TODO: one-hot encode the target values (`keras.utils.to_categorical`)
    Y = tensorflow.keras.utils.to_categorical(Y, num_classes=10)
    return X, Y


def build_model():
    model = Sequential([
        Flatten(input_shape = (28, 28)),
        Dense(512, activation=relu),
        Dense(512, activation=relu),
        Dense(10, activation=softmax)
        # TODO add layers
        # don't forget to sotfmax at the end
        ])
    # Try both:
    # optimizer = SGD(learning_rate = 0.01)
    optimizer = Adam(learning_rate = 1e-4)
    # TODO: choose an appropriate loss function (error metric)
    # https://keras.io/losses/
    loss = 'mse'
    model.compile(loss = loss, optimizer = optimizer,
                  metrics = ['accuracy']) # We care about accuracy
    return model


def train(model, X, Y, num_epochs):
    batch_size = 32
    # You may want to try this:
    callbacks = [
        # EarlyStopping(monitor = 'loss'),
        ]
    history = model.fit(X, Y, epochs = num_epochs,
              callbacks = callbacks,
              verbose = 1,
              validation_split = 0,   # optionally reserve some data for validation
              batch_size = batch_size)
    return history


def evaluate(model, X, Y):
    loss, accuracy = model.evaluate(X, Y)
    return loss, accuracy


def predict(model, X):
    return model.predict(X)


# TODO:
# - gather the data
# - build the model
# - train the model
# - evaluate the model
# - visualise the performance of the trained model
X, Y, X_test, Y_test = get_dataset()
print(X.shape)
print(Y.shape)
print(X_test.shape)
print(Y_test.shape)

X, Y = preprocess_data(X, Y)
X_test, Y_test = preprocess_data(X_test, Y_test)
model = build_model()
history = train(model, X, Y, num_epochs=30)
loss, acc = evaluate(model, X, Y)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('L2 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


