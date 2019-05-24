from utils import *
from model import *
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# Map integer to a word
def word_for_id(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return None

# Generate target given source sequence
def predict_sequence(model, tokenizer, source):
  prediction = model.predict(source, verbose=0)[0]
  integers = [np.argmax(vector) for vector in prediction]
  target = list()
  for i in integers:
    word = word_for_id(i, tokenizer)
    if word is None:
      break
    target.append(word)
  return ' '.join(target)

def evaluate_model(model, sources, raw_dataset):
  actual = list()
  predicted = list()
  for i, source in enumerate(sources):
    source = source.reshape((1, source.shape[0]))
    translation = predict_sequence(model, eng_tokenizer, source)
    raw_target, raw_src = raw_dataset[i]
    if i < 10:
      print('Src=[{:s}], Target=[{:s}], Predicted=[{:s}]'.format(raw_src, raw_target, translation))
    actual.append(raw_target.split())
    predicted.append(translation.split())

  print('BLEU-1 {:f}'.format(corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))))
  print('BLEU-2 {:f}'.format(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))))
  print('BLEU-3 {:f}'.format(corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))))
  print('BLEU-4 {:f}'.format(corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))))

# Load datasets
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# Create tokenizers
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])

trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])

model = load_model('model.h5')

print('[INFO] Evaluate model on training set...')
evaluate_model(model, trainX, train)

print('[INFO] Evaluate model on test set...')
evaluate_model(model, testX, test)
