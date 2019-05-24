import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Embedding
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Encode and pad sequences
def encode_sequences(tokenizer, length, lines):
  X = tokenizer.texts_to_sequences(lines)
  X = pad_sequences(X, maxlen=length, padding='post')
  return X

# One-hot encode the output
def encode_output(sequences, vocab_size):
  ylist = list()
  for seq in sequences:
    encoded = to_categorical(seq, num_classes=vocab_size)
    ylist.append(encoded)
  y = np.array(ylist)
  y = y.reshape((sequences.shape[0], sequences.shape[1], vocab_size))
  return y

def create_tokenizer(lines):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

def define_checkpoint():
  checkpoint = ModelCheckpoint('model.h5',
                                monitor='val_loss',
                                verbose=1,
                                save_best_only=True,
                                mode='min')
  return checkpoint

def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
  model = Sequential()
  model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
  model.add(LSTM(n_units))
  model.add(RepeatVector(tar_timesteps))
  model.add(LSTM(n_units, return_sequences=True))
  model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
  model.summary()
  return model
