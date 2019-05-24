# Simple one-word in, one-word out framing for word-based neural language model

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

def define_model(vocab_size, max_len):
  model = Sequential()
  model.add(Embedding(vocab_size, 10, input_length=max_len-1))
  model.add(LSTM(50))
  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  model.summary()
  return model

def generate_seq(model, tokenizer, max_len, seed_text, n_words):
  in_text = seed_text

  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = pad_sequences([encoded], maxlen=max_len, padding='pre')
    yhat = model.predict_classes(encoded, verbose=0)
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    in_text += ' ' + out_word

  return in_text


data = """
Jack and Jill went up the hill\n
To fetch a pail of water\n
Jack fell down and broke his crown\n
And Jill came tumbling after\n
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]
print(encoded)

vocab_size = len(tokenizer.word_index) + 1
print('Vocab size is: {:d}'.format(vocab_size))

sequences = list()
for i in range(2, len(encoded)):
  seq = encoded[i-2:i+1]
  sequences.append(seq)
print('Sequences length: {:d}'.format(len(sequences)))

max_len = max([len(s) for s in sequences])
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')


sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
print(y.shape)

model = define_model(vocab_size, max_len)
model.fit(X, y, epochs=500, verbose=2)

print(generate_seq(model, tokenizer, max_len-1, 'Jack and', 5))
print(generate_seq(model, tokenizer, max_len-1, 'And Jill', 3))
print(generate_seq(model, tokenizer, max_len-1, 'fell down', 5))
print(generate_seq(model, tokenizer, max_len-1, 'pail of', 5))
