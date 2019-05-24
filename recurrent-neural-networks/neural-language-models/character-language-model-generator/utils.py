import numpy as np
from pickle import load
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def load_doc_chars(filename):
  with open(filename, "r") as f:
    data = f.read()
    data = data.lower()
    chars = sorted(list(set(data)))
  return data, chars

def save_doc(lines, fname):
  data = '\n'.join(lines)
  with open(fname, "w") as f:
    f.write(data)

def load_doc(fname):
  with open(fname, "r") as f:
    text = f.read()
  return text

def decode(x, indices_char, calc_argmax=True):
  if calc_argmax:
    x = x.argmax(axis=-1)
  return ''.join(indices_char[x] for x in x)

def print_sample(sample_ix, ix_to_char):
  txt = ''.join(ix_to_char[ix] for ix in sample_ix)
  txt = txt[0].upper() + txt[1:]  # capitalize first character
  print('%s' % (txt, ), end='')

def sample_preds(preds, temperature=1.0):
  # helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def sample(model, char_to_ix, seq_length, n_chars):
  vocab_size = len(char_to_ix)
  x = np.zeros((1, seq_length, vocab_size))
  indices = []
  idx = -1
  counter = 0
  newline_character = char_to_ix['\n']

  while(idx != newline_character and counter != n_chars):
    y = model.predict(x)
    # Only getting the probs for first character since we are only passing in 1 char
    # to get the next char
    res = y[0][0]
    idx = np.random.choice(list(range(vocab_size)), p=res.ravel())
    indices.append(idx)
    # Overwrite the input character as the one corresponding to the sampled index
    x = np.zeros((1, seq_length, vocab_size))
    x[0, 0, idx] = 1
    counter +=1

  if(counter == n_chars):
    indices.append(char_to_ix['\n'])

  return indices
