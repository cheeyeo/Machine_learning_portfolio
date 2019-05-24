from utils import load_doc, save_doc, sample, print_sample
from pickle import dump
import numpy as np
from modelv2 import define_model_v2
from keras.models import load_model
from keras.callbacks import LambdaCallback

def on_epoch_end(epoch, _):
  print()
  if epoch % 10 == 0:
    print('----- Generating text after Epoch: {}'.format(epoch))
    for i in range(7):
      sampled_indices = sample(model, char_to_ix, seq_length=27, n_chars=50)
      print_sample(sampled_indices, ix_to_char)

data = load_doc("data/dinos.txt")
data = data.lower()
chars = sorted(list(set(data)))

char_to_ix = {c:i for i,c in enumerate(chars)}
print(char_to_ix)
ix_to_char = {i:c for i,c in enumerate(chars)}
vocab_size = len(chars)
print('[INFO] Vocab size: {:d}'.format(vocab_size))

# Prepare inputs (we're sweeping from left to right in steps seq_length long)
seq_length = 27
X = list()
Y = list()
sequencesIn = list()
sequencesOut = list()

for i in range(seq_length, len(data)):
  p = i-seq_length
  a = data[p:p+seq_length]
  b = data[p+1:p+seq_length+1]
  x = [char_to_ix[c] for c in a]
  y = [char_to_ix[c] for c in b]
  X.append(x)
  Y.append(y)
  sequencesIn.append(a)
  sequencesOut.append(b)

# save_doc(sequencesIn, 'char_sequences_in.txt')
# save_doc(sequencesOut, 'char_sequences_out.txt')

X = np.array(X)
Y = np.array(Y)
print("[INFO] X shape: {}, Y shape: {}".format(X.shape, Y.shape))

# One-hot encoding of X, Y
Xtrain = np.zeros((X.shape[0], X.shape[1], vocab_size))
for i, x in enumerate(X):
  for t, idx in enumerate(x):
    Xtrain[i, t, idx] = 1

Ytrain = np.zeros((Y.shape[0], Y.shape[1], vocab_size))
for i, y in enumerate(Y):
  for t, idx in enumerate(y):
    Ytrain[i, t, idx] = 1

print("Xtrain shape: {}, Ytrain shape: {}".format(Xtrain.shape, Ytrain.shape))

print('[INFO] Building model...')
model = define_model_v2(seq_length, vocab_size)

print('[INFO] Training model...')
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model.fit(Xtrain, Ytrain,
          epochs=2000,
          batch_size=64,
          callbacks=[print_callback],
          verbose=1)

print('[INFO] Saving model...')
model.save('modelv2.h5')

print('[INFO] Saving dictionaries...')
dump(char_to_ix, open('char_to_ix.pkl', 'wb'))
dump(ix_to_char, open('ix_to_char.pkl', 'wb'))
