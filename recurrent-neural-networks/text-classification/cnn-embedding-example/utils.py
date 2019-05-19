from nltk.corpus import stopwords
import string
import re
from os import listdir
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def load_doc(filename):
  with open(filename, 'r') as f:
    text = f.read()
  return text

def clean_doc(doc):
  tokens = doc.split()
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  tokens = [re_punc.sub('', w) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]
  # filter out short tokens
  tokens = [word for word in tokens if len(word) > 1]
  return tokens

def add_doc_to_vocab(filename, vocab):
  doc = load_doc(filename)
  tokens = clean_doc(doc)
  vocab.update(tokens)

def process_docs(directory, vocab):
  for filename in listdir(directory):
    if filename.startswith('cv9'):
      continue
    path = directory + '/' + filename
    add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
  data = '\n'.join(lines)
  with open(filename, 'w') as f:
    f.write(data)

def clean_doc_with_vocab(doc, vocab):
  tokens = doc.split()
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  tokens = [re_punc.sub('', w) for w in tokens]
  # filter out tokens not in vocab
  tokens = [w for w in tokens if w in vocab]
  tokens = ' '.join(tokens)
  return tokens

def process_docs_for_train(directory, vocab, is_train):
  documents = list()

  for filename in listdir(directory):
    if is_train and filename.startswith('cv9'):
      continue
    if not is_train and filename.startswith('cv9'):
      continue

    path = directory + '/' + filename
    doc = load_doc(path)
    tokens = clean_doc_with_vocab(doc, vocab)
    documents.append(tokens)

  return documents

def load_clean_dataset(vocab, is_train):
  neg = process_docs_for_train('dataset/txt_sentoken/neg', vocab, is_train)
  pos = process_docs_for_train('dataset/txt_sentoken/pos', vocab, is_train)
  docs = neg + pos
  labels = np.array([0 for _ in range(len(neg))] + [1 for _ in range(len(pos))])
  return docs, labels

def predict_sentiment(review, vocab, tokenizer, max_length, model):
  line = clean_doc_with_vocab(review, vocab)
  encoded = tokenizer.texts_to_sequences([line])
  # Need to store the max length somehow...
  padded = pad_sequences(encoded, maxlen=1317, padding='post')

  yhat = model.predict(padded, verbose=0)
  print(yhat)
  percent_pos = yhat[0,0]
  if round(percent_pos) == 0:
    return (1-percent_pos), 'NEGATIVE'

  return percent_pos, 'POSITIVE'

