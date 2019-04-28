#!/usr/bin/env python
from keras.layers import Input, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense
from keras.models import Model, Sequential
from keras.optimizers import SGD
from cifar_utils import plot_history, load_dataset, normalize_data

# Building model
#
# We will be building baseline model using the VGG architecture, which involves stacking small convolutional layers with `3x3` filters followed by max pooling layer.
#
# These layers form a block and blocks repeated with where nos of filters in each block increase as we increase depth of the network.
#
# We will be building VGG-style models with the following configuration:
#
# * VGG 1 ( 1 VGG block )
#
# * VGG 2 ( 2 VGG blocks)
#
# * VGG 3 ( 3 VGG blocks )


def vgg_block1():
    inputs = Input(shape=(32, 32, 3))
    X = Conv2D(32, (3,3), kernel_initializer="he_uniform", padding="same")(inputs)
    X = Activation("relu")(X)
    X = Conv2D(32, (3,3), kernel_initializer="he_uniform", padding="same")(X)
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

def vgg_block2():
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
    X = Flatten()(X)
    X = Dense(128, kernel_initializer="he_uniform")(X)
    X = Activation("relu")(X)
    X = Dense(10)(X)
    final_layer = Activation("softmax")(X)

    model = Model(inputs=inputs, outputs=final_layer)
    opt = SGD(lr=0.001, momentum=0.9)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

    return model

def vgg_block3():
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

print("VGG BLOCK 1 model")
model = vgg_block1()
print(model.summary())
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1)
loss, acc = model.evaluate(testX, testY, verbose=0)

print("VGG 1 block Loss: {:.3f}, Acc: {:.3f}".format((loss), (acc*100)))
plot_history(history, "cifar/vgg_block1")

print()
print("VGG BLOCK 2 model")
model = vgg_block2()
print(model.summary())
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1)
loss, acc = model.evaluate(testX, testY, verbose=0)

print("VGG 2 block Loss: {:.3f}, Acc: {:.3f}".format((loss), (acc*100)))
plot_history(history, "cifar/vgg_block2")

print()
print("VGG BLOCK 3 model")
model = vgg_block3()
print(model.summary())
history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1)
loss, acc = model.evaluate(testX, testY, verbose=0)

print("VGG 3 block Loss: {:.3f}, Acc: {:.3f}".format((loss), (acc*100)))
plot_history(history, "cifar/vgg_block3")
