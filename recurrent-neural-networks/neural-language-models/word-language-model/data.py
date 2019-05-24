from utils import *

doc = load_doc('data/republic_clean.txt')
print(doc[:200])

tokens = clean_doc(doc)
print(tokens[:200])
print("Total tokens: {:d}".format(len(tokens)))
print("Unique tokens: {:d}".format(len(set(tokens))))

length = 51
sequences = list()
for i in range(length, len(tokens)):
  seq = tokens[i-length:i]
  line = ' '.join(seq)
  sequences.append(line)
print('Total sequences: {:d}'.format(len(sequences)))

save_doc(sequences, 'republic_sequences.txt')
