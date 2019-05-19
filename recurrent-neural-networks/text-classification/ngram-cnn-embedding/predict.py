from utils import predict_sentiment, load_dataset
from keras.models import load_model

model = load_model('model.h5')
tokenizer = load_dataset('tokenizer.pkl')

trainX, _ = load_dataset('data/train.pkl')
max_length = max([len(s.split()) for s in trainX])
print(max_length)

with open('data/txt_sentoken/neg/cv999_14636.txt') as f:
  review = f.read()

percent, sentiment = predict_sentiment(review, tokenizer, max_length, model)
print("Review: [{}]\nSentiment: {} {:.3f}%".format(review, sentiment, percent*100))

with open('data/txt_sentoken/pos/cv999_13106.txt') as f:
  review = f.read()

percent, sentiment = predict_sentiment(review, tokenizer, max_length, model)
print("Review: [{}]\nSentiment: {} {:.3f}%".format(review, sentiment, percent*100))

