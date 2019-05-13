import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding

docs = [
  "Well done!",
  "Good work",
  "Great effort",
  "nice work",
  "Excellent!",
  "Weak",
  "Poor effort!",
  "not good",
  "poor work",
  "Could have done better."
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

t = Tokenizer()
t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

# pad docs to max length of 4
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding="post")
print(padded_docs)

# load embedding
embeddings_index = {}
with open('data/glove.6B.100d.txt', mode='rt', encoding='utf-8') as f:
  for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print("Loaded word embeddings: {}".format(len(embeddings_index)))

# Create weight matrix for words in training docs
# 100 as we are using a 100 dimensional embedding
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
  embedding_vector = embeddings_index[word]
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector

# Define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=4, weights=[embedding_matrix], trainable=False))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(padded_docs, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print("Accuracy: {:.3f}".format(accuracy*100))
