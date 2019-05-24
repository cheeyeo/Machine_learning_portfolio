import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
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

vocab_size = len(tokenizer.word_index) + 1
print('Vocab size is: {:d}'.format(vocab_size))

# line based sequences
sequences = list()
for line in data.split('\n'):
  encoded = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(encoded)):
    seq = encoded[:i+1]
    sequences.append(seq)

print(sequences)
print("Total sequences: {:d}".format(len(sequences)))

# pad sequences
max_len = max([len(x) for x in sequences])
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences)
print('Max seq length: {:d}'.format(max_len))

sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

model = define_model(vocab_size, max_len)
model.fit(X, y, epochs=500, verbose=2)

print(generate_seq(model, tokenizer, max_len-1, 'Jack', 4))
print(generate_seq(model, tokenizer, max_len-1, 'Jill', 4))
