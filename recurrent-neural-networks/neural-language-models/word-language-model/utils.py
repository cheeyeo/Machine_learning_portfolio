import re
import string

def load_doc(filename):
  with open(filename, 'r') as f:
    text = f.read()
  return text

def clean_doc(doc):
  doc = doc.replace('--', ' ')
  tokens = doc.split()
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  tokens = [re_punc.sub('', w) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens

def save_doc(lines, filename):
  data = '\n'.join(lines)
  with open(filename, 'w') as f:
    f.write(data)
