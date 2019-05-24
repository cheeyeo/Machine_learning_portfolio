import string
import re
from pickle import dump, load
from unicodedata import normalize
import numpy as np

# Load file as blob of text to preserve the unicode chars
def load_doc(filename):
  with open(filename, mode='rt', encoding='utf-8') as f:
    text = f.read()
  return text

def to_pairs(doc):
  lines = doc.strip().split('\n')
  pairs = [line.split('\t') for line in  lines]
  return pairs

# clean a list of lines
def clean_pairs(lines):
  cleaned = list()
  # prepare regex for char filtering
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  re_print = re.compile('[^%s]' % re.escape(string.printable))
  for pair in lines:
    clean_pair = list()
    for line in pair:
      # normalize unicode characters
      line = normalize('NFD', line).encode('ascii', 'ignore')
      line = line.decode('UTF-8')
      # tokenize on white space
      line = line.split()
      # convert to lowercase
      line = [word.lower() for word in line]
      # remove punctuation from each token
      line = [re_punc.sub('', w) for w in line]
      # remove non-printable chars form each token
      line = [re_print.sub('', w) for w in line]
      # remove tokens with numbers in them
      line = [word for word in line if word.isalpha()]
      # store as string
      clean_pair.append(' '.join(line))
    cleaned.append(clean_pair)
  return np.array(cleaned)

def save_clean_data(data, filename):
  dump(data, open(filename, 'wb'))
  print('[INFO] Saved {}'.format(filename))

def load_clean_sentences(filename):
  return(load(open(filename, 'rb')))

def max_length(lines):
  return max(len(line.split()) for line in lines)
