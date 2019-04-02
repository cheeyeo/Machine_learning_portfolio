from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os
import matplotlib.pyplot as plt

np.random.seed(7) # for reproducibility

absdir = os.path.dirname(os.path.realpath('__file__'))
datapath = "data/pima-indians-diabetes.csv"
dataset = np.loadtxt(os.path.join(absdir, datapath), delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, validation_split=0.25, epochs=150, batch_size=10)
# print(history.history)

scores = model.evaluate(X, y)
# print(scores)
# print(model.metrics_names)
print("{}: {:.2f}%".format(model.metrics_names[0].capitalize(), scores[0]*100))
print("{}: {:.2f}%".format(model.metrics_names[1].capitalize(), scores[1]*100))

# Plot training loss and accuracy
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['acc'], label='Accuracy')
plt.title('Training loss vs accuracy')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()
