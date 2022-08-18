from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np


def fizzbuzz(i):
    if i % 15 == 0:
        return [0, 0, 0, 1]
    elif i % 5 == 0:
        return [0, 0, 1, 0]
    elif i % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def bin(i, num_digits):
    return [i >> d & 1 for d in range(num_digits)]


x_train = np.array([bin(i, 7) for i in range(1, 101)]).astype(np.float32)
y_train = np.array([fizzbuzz(i) for i in range(1, 101)]).astype(np.float32)
model = Sequential()
model.add(Dense(64, activation='tanh', input_dim=7))
model.add(Dense(4, activation='softmax', input_dim=64))
model.compile(
    loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3600, batch_size=64)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_tflite = converter.convert()

with open('fizzbuzz_model.tflite', 'wb') as f:
    f.write(model_tflite)
