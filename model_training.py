import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import center_by_mass


mnist = tf.keras.datasets.mnist


(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = np.array([cv2.threshold(img,1,1,cv2.THRESH_BINARY)[1] for img in x_train])
x_test = np.array([cv2.threshold(img,1,1,cv2.THRESH_BINARY)[1] for img in x_test])
print('Preprocessing!')
x_train = np.array([center_by_mass(image) for image in x_train])
x_test = np.array([center_by_mass(image) for image in x_test])

print('done preprocessing')
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.compile(optimizer = "adam", loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4)

model.save("my_first.model")

