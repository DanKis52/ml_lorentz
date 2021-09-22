import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # removing GPU warnings
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import lorenz


# training dataset
training_input = lorenz.training_in
training_output = lorenz.training_out

#  testing dataset
testing_input = lorenz.testing_in
testing_output = lorenz.testing_out

# initialize NN
model = Sequential()
model.add(Dense(units=128, input_shape=(20,)))  # input layer
model.add(Dense(units=64))  # hidden layers
model.add(Dense(units=64))
model.add(Dense(units=1))  # output layer
opt = SGD(momentum=0.9, learning_rate=0.000001)  # used a modified version of GD called stochastic gradient descent
model.compile(loss='mean_squared_error', optimizer=opt)

history = model.fit(training_input, training_output, epochs=10, verbose=True)  # learning

plt.plot(history.history['loss'])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()

x_predicted = model.predict(lorenz.testing_in)
plt.plot(testing_output)
plt.plot(x_predicted,'--')
plt.legend(['Real output', 'Predicted output'])
plt.show()
