# Process the docs and save the tokens into
# a vocab file
from utils import process_docs
from utils import save_list
from utils import load_doc
from collections import Counter

vocab = Counter()
process_docs('dataset/txt_sentoken/neg', vocab)
process_docs('dataset/txt_sentoken/pos', vocab)
print("[INFO] Vocab length: {}".format(len(vocab)))

min_occurences = 2
tokens = [k for k,c in vocab.items() if c >= min_occurences]
print("[INFO] Vocab length after processing: {}".format(len(tokens)))
save_list(tokens, 'vocab.txt')

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
