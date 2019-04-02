# MLP with automatic verification dataset
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os

np.random.seed(7)

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/pima-indians-diabetes.csv"
dataset = np.loadtxt(os.path.join(absdir, datapath), delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X, y, epochs=150, batch_size=10, validation_split=0.33)
