from nltk.corpus import stopwords
import string
import re
from os import listdir
import numpy as np
from pickle import dump
from pickle import load
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
  with open(filename, 'r') as f:
    text = f.read()
  return text

def clean_doc(doc):
  tokens = doc.split()
  punc = re.compile('[%s]' % re.escape(string.punctuation))
  # remove punctuations
  tokens = [punc.sub('', w) for w in tokens]
  # remove non-alphabetic tokens
  tokens = [w for w in tokens if w.isalpha()]
  # remove stop words
  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]
  # remove short tokens
  tokens = [w for w in tokens if len(w) > 1]
  tokens = ' '.join(tokens)
  return tokens

def process_docs(directory, is_train):
  documents = list()

  for filename in listdir(directory):
    # skip any reviews in the test set
    if is_train and filename.startswith('cv9'):
      continue
    if not is_train and not filename.startswith('cv9'):
      continue

    path = directory + '/' + filename
    # load the doc
    doc = load_doc(path)
    # clean doc
    tokens = clean_doc(doc)
    documents.append(tokens)
  return documents

def load_clean_dataset(is_train):
  neg = process_docs('data/txt_sentoken/neg', is_train)
  pos = process_docs('data/txt_sentoken/pos', is_train)
  docs = neg + pos
  labels = np.array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
  return docs, labels

def save_dataset(dataset, filename):
  dump(dataset, open(filename, 'wb'))
  print("Saved {}".format(filename))

def load_dataset(filename):
  return load(open(filename, 'rb'))

def predict_sentiment(review, tokenizer, max_length, model):
  line = clean_doc(review)
  encoded = tokenizer.texts_to_sequences([line])
  padded = pad_sequences(encoded, maxlen=max_length, padding='post')

  yhat = model.predict([padded, padded, padded], verbose=0)
  print(yhat)
  percent_pos = yhat[0,0]
  if round(percent_pos) == 0:
    return (1-percent_pos), 'NEGATIVE'

  return percent_pos, 'POSITIVE'
