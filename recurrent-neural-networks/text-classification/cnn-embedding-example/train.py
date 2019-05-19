# Word embedding + CNN example for sentiment classification
# Trains the embedding as part of learning process
import pickle
from utils import load_doc, load_clean_dataset
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import load_model

def base_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 100, input_length=max_length))
  model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

trainX, ytrain = load_clean_dataset(vocab, True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(trainX)
vocab_size = len(tokenizer.word_index) + 1
print("[INFO] Vocab size: {:d}".format(vocab_size))

max_length = max([len(s.split()) for s in trainX])
print("[INFO] Max length: {:d}".format(max_length))

encoded = tokenizer.texts_to_sequences(trainX)
Xtrain = pad_sequences(encoded, maxlen=max_length, padding='post')
print("[INFO] Xtrain shape: {}, ytrain shape: {}".format(Xtrain.shape, ytrain.shape))

model = base_model(vocab_size, max_length)
model.summary()

model.fit(Xtrain, ytrain, epochs=10, verbose=2)

model.save('model.h5')
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

testX, ytest = load_clean_dataset(vocab, False)
encoded = tokenizer.texts_to_sequences(testX)
Xtest = pad_sequences(encoded, maxlen=max_length, padding='post')

model = load_model('model.h5')
_, acc = model.evaluate(Xtest, ytest, verbose=0)
print("[INFO]: Test accuracy: {:.2f}".format(acc*100))
