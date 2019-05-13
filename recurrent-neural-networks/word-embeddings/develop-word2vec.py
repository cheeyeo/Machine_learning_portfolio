from gensim.models import Word2Vec

# define training data as sentences tokenized
training_data = [
  ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
  ['this', 'is', 'the', 'second', 'sentence'],
  ['yet', 'another', 'sentence'],
  ['one', 'more', 'sentence'],
  ['and', 'the', 'final', 'sentence']
]

model = Word2Vec(training_data, min_count=1)
print(model)

# view vocab
words = list(model.wv.vocab)
print(words)

# view vector for one word
print(model['sentence'])
print(model['sentence'].shape)

# save model
model.save('model.bin')

# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
