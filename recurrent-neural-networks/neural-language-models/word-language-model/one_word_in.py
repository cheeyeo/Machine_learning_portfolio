# Simple one-word in, one-word out framing for word-based neural language model

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

def define_model():
  model = Sequential()
  model.add(Embedding(vocab_size, 10, input_length=1))
  model.add(LSTM(50))
  model.add(Dense(vocab_size, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  model.summary()
  return model

def generate_seq(model, tokenizer, seed_text, n_words):
  in_text, result = seed_text, seed_text
  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = np.array(encoded)
    yhat = model.predict_classes(encoded, verbose=0)
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    in_text = out_word
    result = result + ' ' + out_word

  return result


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
for i in range(1, len(encoded)):
  seq = encoded[i-1:i+1]
  sequences.append(seq)
print('Sequences length: {:d}'.format(len(sequences)))

sequences = np.array(sequences)
X, y = sequences[:,0], sequences[:,1]
y = to_categorical(y, num_classes=vocab_size)
print(y.shape)

model = define_model()
model.fit(X, y, epochs=500, verbose=2)

print(generate_seq(model, tokenizer, 'Jack', 6))
