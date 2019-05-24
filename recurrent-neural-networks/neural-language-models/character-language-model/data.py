from utils import load_doc, save_sequences
import numpy as np

raw_text = load_doc('data/rhyme.txt')

# remove newlines and re-join characters into one long sequence
tokens = raw_text.split()
raw_text = ' '.join(tokens)

# Create sequences. Each input sequence 11 chars long; 10 chars as input, 1 for output
length = 10
sequences = list()

for i in range(length, len(raw_text)):
  seq = raw_text[i-length:i+1]
  sequences.append(seq)

print("Total sequences: {:d}".format(len(sequences)))

save_sequences(sequences, 'char_sequences.txt')
