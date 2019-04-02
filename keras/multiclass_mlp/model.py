# Example of multiclass MLP using the iris dataset
import os
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

def base_model():
  model = Sequential()
  model.add(Dense(4, input_dim=4, kernel_initializer="normal", activation="relu"))
  model.add(Dense(3, kernel_initializer="normal", activation="softmax")) # use softmax??
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
  return model

seed = 7
np.random.seed(seed)

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/iris.csv"
dataframe = read_csv(os.path.join(absdir, datapath), header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
y = dataset[:, 4]

# encode class values as integers then one-hot encode
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_y)
print(dummy_y[0])

estimator = KerasClassifier(build_fn=base_model, epochs=200, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: {:.2f}% Std Dev: {:.2f}%".format(results.mean()*100, results.std()*100))
