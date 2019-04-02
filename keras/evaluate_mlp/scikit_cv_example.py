# MLP with 10-Fold cross validation via sklearn integration
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import os

def create_model():
  model = Sequential()
  model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
  model.add(Dense(8, kernel_initializer="uniform", activation="relu"))
  model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
  return model

seed = 7
np.random.seed(seed)

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/pima-indians-diabetes.csv"
dataset = np.loadtxt(os.path.join(absdir, datapath), delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=1)
# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())
