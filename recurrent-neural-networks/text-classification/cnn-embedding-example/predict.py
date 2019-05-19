from utils import load_doc, clean_doc_with_vocab, predict_sentiment
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import load

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())

model = load_model('model.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

review = 'Everyone will enjoy this film. I love it, recommended!'
percent, sentiment = predict_sentiment(review, vocab, tokenizer, 1317, model)
print("Review: [{}]\nSentiment: {} {:.3f}%".format(review, sentiment, percent*100))

# For negative reviews, it seems that the length of a review does have an effect on its classification. Shorter reviews like below tend to result in 60%+ scores, which cause it to be positive. Hence test with actual negative review from the test dataset for confirmation
# review = 'This is a bad movie. Do not watch it. It sucks.'
with open('dataset/txt_sentoken/neg/cv999_14636.txt') as f:
  review = f.read()
percent, sentiment = predict_sentiment(review, vocab, tokenizer, 1317, model)
print("Review: [{}]\nSentiment: {} {:.3f}%".format(review, sentiment, percent*100))

