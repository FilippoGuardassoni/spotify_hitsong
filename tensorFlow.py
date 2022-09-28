import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Import dataset
df = pd.read_csv('/Users/amministratore/Documents/Data Science/First Year/Second Semester/Statistical Learning, Deep Learning and Artificial Intelligence/Statistical Learning/Individual Project/my data/songs.csv')
print(df.dtypes)

# Change rel_date type to datetime
df["rel_date"] = pd.to_datetime(df["rel_date"])
print(df.dtypes)

# Filter songs from 2007
df = df[(df["rel_date"] >= "2007-01-01")]
print(df.describe())

# Drop unneeded variables
idx = np.r_[0, 1, 4, 6, 20:25]
df.drop(df.columns[[idx]], axis=1, inplace=True)
print(df.head())

# Scale columns non ranging from 0 to 1
scaler = MinMaxScaler()
df[["pop_artist", "tot_followers", "avail_mark", "pop_track", "duration_ms", "loudness", "tempo", "time_signature"]] = scaler.fit_transform(df[["pop_artist", "tot_followers", "avail_mark", "pop_track", "duration_ms", "loudness", "tempo", "time_signature"]])
print(df.head())

# Splitting data into a training set and testing set
inp = df.iloc[:, :-1]
out = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(inp, out, test_size=0.20, random_state=123, shuffle=True, stratify=out)

# LOGISTIC REGRESSION
# HANDWRITTEN CODE
# Visualize used activation function
def sigmoid(z):
    return 1 / (1 + np.exp(- z))

plt.plot(np.arange(-5, 5, 0.1), sigmoid(np.arange(-5, 5, 0.1)))
plt.title('Visualization of the Sigmoid Function')

plt.show()

# Dataset parameters
num_features = 15
num_classes = 2

# Training parameters
learning_rate = 0.01
training_steps = 500
batch_size = 256
display_step = 10

# Convert to float32
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

# Flatten dataframe to 1-D array
x_train, x_test = np.reshape(x_train, (-1, num_features)), np.reshape(x_test, (-1, num_features))

# Batch the data
train_data=tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data=train_data.repeat().batch(batch_size).prefetch(1)

# Trainable Variable Weights
W = tf.Variable(tf.zeros([num_features, num_classes]), name="weight")

# Trainable Variable Bias
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# Logistic regression (Wx + b).
def logistic_regression(x):
    return tf.nn.sigmoid(tf.matmul(x, W) + b)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Stochastic gradient descent optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Optimization process.
def grad(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)
    # Compute gradients
    gradients = g.gradient(loss, [W, b])
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))


# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):

    # Run the optimization to update W and b values.
    grad(batch_x, batch_y)
    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

# Test model on validation set.
pred = logistic_regression(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

# LOGISTIC REGRESSION USING KERAS
# Define the model
lr = Sequential()
lr.add(Dense(9, activation='relu', kernel_initializer='he_normal', input_shape=(num_features,), kernel_regularizer='l2'))
lr.add(Dense(1, activation='sigmoid'))

# Compile the model
lr.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train and Test the model
train_history = lr.fit(x_train, y_train, batch_size=256, epochs=100)
test_history = lr.fit(x_test, y_test, batch_size=256, epochs=100)

loss_lr, acc_lr = lr.evaluate(x_test, y_test)
print(f"Loss is {loss_lr},\nAccuracy is {acc_lr * 100}")

# summarize history for accuracy
plt.plot(train_history.history['binary_accuracy'])
plt.plot(test_history.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(train_history.history['loss'])
plt.plot(test_history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# MULTILAYER PERCEPTRON (MLP) NEURAL NETWORK
'''
    A Feed-forward neural network with:
    - 15 inputs
    - 2 hidden layers (hl1: 15 inp + 1 out/2 = 8 neurons, hl2 = hl1/2 = 4)
    - 1 output layer

    '''
# define model
mlp_1 = Sequential()
mlp_1.add(Dense(8, activation='relu', kernel_initializer='he_normal', input_shape=(num_features,)))
mlp_1.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
mlp_1.add(Dense(1, activation='sigmoid'))
# compile the model
sgd = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
mlp_1.compile(optimizer=sgd, loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
# fit the model
train_history_ffn = mlp_1.fit(x_train, y_train, epochs=500, batch_size=256)
loss_ffn, acc_ffn = mlp_1.evaluate(x_train, y_train)
print("Train Loss: %f, Train Accuracy: %f" % (loss_ffn, acc_ffn))

# evaluate the model
test_history_ffn = mlp_1.fit(x_test, y_test, epochs=500, batch_size=256)
val_loss_ffn, val_acc_ffn = mlp_1.evaluate(x_test, y_test)
print("Test Loss: %f, Test Accuracy: %f" % (val_loss_ffn, val_acc_ffn))

# summarize history for accuracy
# plot the training loss and accuracy
plt.plot(train_history_ffn.history['binary_accuracy'])
plt.plot(test_history_ffn.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(train_history_ffn.history['loss'])
plt.plot(test_history_ffn.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# MULTILAYER PERCEPTRON (MLP) NEURAL NETWORK
'''
    A Feed-forward neural network with:
    - 15 inputs
    - 4 hidden layers
    - 1 output layer

    '''
# define model
mlp_2 = Sequential()
mlp_2.add(Dense(15, activation='relu', kernel_initializer='he_normal', input_shape=(num_features,)))
mlp_2.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
mlp_2.add(Dense(3, activation='relu', kernel_initializer='he_normal'))
mlp_2.add(Dense(1, activation='sigmoid'))
# compile the model
sgd = tf.keras.optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
mlp_2.compile(optimizer=sgd, loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
# fit the model
train_history_ffn2 = mlp_2.fit(x_train, y_train, epochs=500, batch_size=256)
loss_ffn2, acc_ffn2 = mlp_2.evaluate(x_train, y_train)
print("Train Loss: %f, Train Accuracy: %f" % (loss_ffn2, acc_ffn2))

# evaluate the model
test_history_ffn2 = mlp_2.fit(x_test, y_test, epochs=500, batch_size=256)
val_loss_ffn2, val_acc_ffn2 = mlp_2.evaluate(x_test, y_test)
print("Test Loss: %f, Test Accuracy: %f" % (val_loss_ffn2, val_acc_ffn2))

# summarize history for accuracy
# plot the training loss and accuracy
plt.plot(train_history_ffn2.history['binary_accuracy'])
plt.plot(test_history_ffn2.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(train_history_ffn2.history['loss'])
plt.plot(test_history_ffn2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
