#!/usr/bin/env python

from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from cifar_utils import plot_history, load_dataset, normalize_data

def vgg_block3_data_augmentation():
    inputs = Input(shape=(32, 32, 3))
    X = Conv2D(32, (3,3), kernel_initializer="he_uniform", padding="same")(inputs)
    X = Activation("relu")(X)
    X = Conv2D(32, (3,3), kernel_initializer="he_uniform", padding="same")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(64, (3,3), kernel_initializer="he_uniform", padding="same")(inputs)
    X = Activation("relu")(X)
    X = Conv2D(64, (3,3), kernel_initializer="he_uniform", padding="same")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)
    X = Conv2D(128, (3,3), kernel_initializer="he_uniform", padding="same")(inputs)
    X = Activation("relu")(X)
    X = Conv2D(128, (3,3), kernel_initializer="he_uniform", padding="same")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D((2, 2))(X)
    X = Flatten()(X)
    X = Dense(128, kernel_initializer="he_uniform")(X)
    X = Activation("relu")(X)
    X = Dense(10)(X)
    final_layer = Activation("softmax")(X)

    model = Model(inputs=inputs, outputs=final_layer)
    opt = SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

trainX, trainY, testX, testY = load_dataset()
trainX, testX = normalize_data(trainX, testX)

print("VGG 3 Blocks with Data Augmentation")
model = vgg_block3_data_augmentation()
print(model.summary())

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

it_train = datagen.flow(trainX, trainY, batch_size=64)
steps = int(trainX.shape[0] / 64)

history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=100, validation_data=(testX, testY))

_, acc = model.evaluate(testX, testY, verbose=0)

print("VGG 3 w/Augmentation Acc: {:.3f}".format(acc*100))

plot_history(history, "vgg_block3_augmentation")
