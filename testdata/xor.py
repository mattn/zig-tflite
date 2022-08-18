from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import numpy as np

x_train = np.array([[0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1]]).astype(np.float32)

y_train = np.array([0,
                    1,
                    1,
                    0]).astype(np.float32)


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(2,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adam', metrics=["mae"])
model.fit(x_train, y_train, epochs=500, batch_size=4)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_tflite = converter.convert()

with open('xor_model.tflite', 'wb') as f:
    f.write(model_tflite)
