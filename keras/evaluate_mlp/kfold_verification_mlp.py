# MLP with K-Fold cross validation
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

def get_model():
  model = Sequential()
  model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
  model.add(Dense(8, activation="relu"))
  model.add(Dense(1, activation="sigmoid"))
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
  return model

seed = 7
np.random.seed(seed)

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/pima-indians-diabetes.csv"
dataset = np.loadtxt(os.path.join(absdir, datapath), delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]

# define 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
  model = get_model()
  model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0)
  scores = model.evaluate(X[test], y[test], verbose=0)
  print("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1]*100)

print("{:.2f}% (+/- {:.2f})%".format(np.mean(cvscores), np.std(cvscores)))
