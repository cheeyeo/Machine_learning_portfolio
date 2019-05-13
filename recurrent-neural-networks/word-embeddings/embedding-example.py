from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding

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

vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding="post")
print(padded_docs)

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(padded_docs, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)

print("Accuracy: {:.3f}".format(accuracy*100))
