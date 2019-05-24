from utils import load_doc
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

def define_model(X, vocab_size):
  model = Sequential()
  model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  model.summary()
  return model

raw_text = load_doc('char_sequences.txt')
lines = raw_text.split('\n')

chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
vocab_size = len(mapping)
print("Vocab size: {:d}".format(vocab_size))

sequences = list()
for line in lines:
  encoded_seq = [mapping[char] for char in line]
  sequences.append(encoded_seq)

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:, -1]

sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = np.array(sequences)
y = to_categorical(y, num_classes=vocab_size)
print(X.shape)
print(y.shape)

model = define_model(X, vocab_size)
model.fit(X, y, epochs=100, verbose=2)
model.save('model.h5')

dump(mapping, open('mapping.pkl', 'wb'))
