def load_doc(filename):
  with open(filename, 'r') as f:
    text = f.read()

  return text

def save_sequences(seqs, filename):
  data = '\n'.join(seqs)
  with open(filename, 'w') as f:
    f.write(data)
