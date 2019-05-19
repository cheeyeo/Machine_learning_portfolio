# Implementation of word embedding + conv for text classification
# conv layer with different filter sizes
# https://github.com/keras-team/keras/issues/6547
# chapter 15 of deep learning with nlp book

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten, Embedding, Input, Activation, Dropout
from keras.layers.merge import concatenate
from keras.utils import plot_model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from utils import load_dataset, save_dataset
import numpy as np

def define_model(length, vocab_size):
  # channel 1
  inputs1 = Input(shape=(length,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=32, kernel_size=4)(embedding1)
  activation1 = Activation('relu')(conv1)
  drop1 = Dropout(0.5)(activation1)
  pool1 = MaxPooling1D(pool_size=2)(drop1)
  flat1 = Flatten()(pool1)

  # channel 2
  inputs2 = Input(shape=(length,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=32, kernel_size=6)(embedding2)
  activation2 = Activation('relu')(conv2)
  drop2 = Dropout(0.5)(activation2)
  pool2 = MaxPooling1D(pool_size=2)(drop2)
  flat2 = Flatten()(pool2)

  # channel3
  inputs3 = Input(shape=(length,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=32, kernel_size=8)(embedding3)
  activation3 = Activation('relu')(conv3)
  drop3 = Dropout(0.5)(activation3)
  pool3 = MaxPooling1D(pool_size=2)(drop3)
  flat3 = Flatten()(pool3)

  merged = concatenate([flat1, flat2, flat3])
  dense1 = Dense(10, activation='relu')(merged)
  output = Dense(1, activation='sigmoid')(dense1)

  model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model

trainX, Y = load_dataset('data/train.pkl')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainX)

length = max([len(s.split()) for s in trainX])
vocab_size = len(tokenizer.word_index) + 1
print("Max document length: {:d}".format(length))
print("Vocab size: {:d}".format(vocab_size))

# Encode and pad the data
encoded = tokenizer.texts_to_sequences(trainX)
X = pad_sequences(encoded, maxlen=length, padding='post')

model = define_model(length, vocab_size)
# plot_model(model, to_file='model.png')

model.fit([X, X, X], Y, batch_size=16, epochs=7)

model.save('model.h5')

save_dataset(tokenizer, 'tokenizer.pkl')
