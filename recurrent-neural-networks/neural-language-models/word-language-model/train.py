from utils import load_doc
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from pickle import dump

def define_model(vocab_size, seq_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 50, input_length=seq_length))
  model.add(LSTM(100, return_sequences=True))
  model.add(LSTM(100))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model

doc = load_doc('republic_sequences.txt')
lines = doc.split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1
print('Vocab size is: {:d}'.format(vocab_size))

sequences = np.array(sequences)
X = sequences[:,:-1]
Y = sequences[:,-1]
Y = to_categorical(Y, num_classes=vocab_size)
seq_length = X.shape[1]

model = define_model(vocab_size, seq_length)

model.fit(X, Y, epochs=100, batch_size=64, verbose=1)

model.save('model.h5')

dump(tokenizer, open('tokenizer.pkl', 'wb'))
