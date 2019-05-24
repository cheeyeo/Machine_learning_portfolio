from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.optimizers import Adam

# Defines a seq2seq model
def define_model_v2(seq_len, vocab_size):
  model = Sequential()
  model.add(LSTM(100, input_shape=(seq_len, vocab_size), return_sequences=True))
  model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
  opt = Adam(lr=0.01, clipvalue=5.0)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
  model.summary()
  return model
