# MLP with manual verification dataset using scikit-learn to split
# dataset into train and test sets
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import os

seed = 7
np.random.seed(seed)

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/pima-indians-diabetes.csv"
dataset = np.loadtxt(os.path.join(absdir, datapath), delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))
