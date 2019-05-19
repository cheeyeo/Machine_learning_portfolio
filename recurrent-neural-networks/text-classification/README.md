# Text Classification

Examples on using word embeddings with convolutional networks to build document
classifiers.

The convolutional network learns to extract features from the word embedding
by inspecting a window of words.

The example are as follows:

* Single convolutional layer ( cnn-embedding-example )
  Uses a word embedding + 1D conv layer

* Multiple convolutional layers ( ngram-cnn-embedding )

  Use mulitple word embedding + 1D conv layers with 3 different filter sizes,
  which represent different ngrams/groups of words read from the input document
