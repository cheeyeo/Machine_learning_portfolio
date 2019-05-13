from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# load model
model = Word2Vec.load('model.bin')

# get vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# Create scatter plot of projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
  plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
