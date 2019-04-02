# MLP with 10-Fold cross validation via sklearn integration
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import os

def create_model(optimizer="rmsprop", kernel_initializer="glorot_uniform"):
  model = Sequential()
  model.add(Dense(12, input_dim=8, kernel_initializer=kernel_initializer, activation="relu"))
  model.add(Dense(8, kernel_initializer=kernel_initializer, activation="relu"))
  model.add(Dense(1, kernel_initializer=kernel_initializer, activation="sigmoid"))
  model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
  return model

seed = 7
np.random.seed(seed)

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/pima-indians-diabetes.csv"
dataset = np.loadtxt(os.path.join(absdir, datapath), delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]

model = KerasClassifier(build_fn=create_model, verbose=1)

optimizers = ["rmsprop", "adam"]
init = ["glorot_uniform", "normal", "uniform"]
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, kernel_initializer=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, y)
print("Best {:f} using {}".format(grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stddev, param in zip(means, stds, params):
  print("{:f} ({:f}%) with {:r}".format(mean, stddev, param))
