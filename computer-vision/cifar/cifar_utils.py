import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np

def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one-hot encode labels
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY

def normalize_data(trainX, testX):
    train_norm = trainX.astype("float32")
    test_norm = testX.astype("float32")
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm

def plot_history(history, filename="vgg_block1"):
  plt.style.use("ggplot")
  (fig, ax) = plt.subplots(2, 1, figsize=(13, 13))

  # plot loss
  ax[0].set_title("Categorical Loss")
  ax[0].set_xlabel("Epochs #")
  ax[0].set_ylabel("Loss")
  ax[0].plot(history.history["loss"], label="loss", color="blue")
  ax[0].plot(history.history["val_loss"], label="val_loss", color="red")
  ax[0].legend(["loss", "val_loss"])

  # plot accuracy
  ax[1].set_title("Classification Accuracy")
  ax[1].set_xlabel("Epochs #")
  ax[1].set_ylabel("Accuracy")
  ax[1].plot(history.history["acc"], label="acc", color="blue")
  ax[1].plot(history.history["val_acc"], label="val_acc", color="red")
  ax[1].legend(["acc", "val_acc"])

  # save plot to file
  plt.savefig(filename + '_plot.png')
  plt.close()
