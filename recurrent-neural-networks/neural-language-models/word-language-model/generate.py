from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pickle import load
from random import randint
from utils import load_doc

def generate(model, tokenizer, seq_length, seed_text, n_chars):
  result = list()
  in_text = seed_text

  for _ in range(n_chars):
    encoded = tokenizer.texts_to_sequences([input_text])[0]
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    yhat = model.predict_classes(encoded, verbose=0)

    out_word=''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    in_text += ' ' + out_word
    result.append(out_word)
  return ' '.join(result)

doc = load_doc('republic_sequences.txt')
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

model = load_model('model.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

seed_text = lines[randint(0, len(lines))]
print('{}\n'.format(seed_text))

generated = generate(model, tokenizer, seq_length, seed_text, 50)
print(generated)
