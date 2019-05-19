import numpy as np

def read_glove_vecs(filename):
  with open(filename, 'r') as f:
    words = set()
    word_to_vec_map = {}

    for line in f:
      line = line.strip().split()
      curr_word = line[0]
      words.add(curr_word)
      word_to_vec_map[curr_word] = np.array(line[1:], dtype='float64')

    return words, word_to_vec_map

def cosine_similarity(u, v):
  """
  Measures how similar two words are by measuring degree of similarity between
  2 embedding vectors for the words
  """
  distance = 0.0
  dot = np.dot(u, v)
  norm_u = np.sqrt(np.sum([u**2 for u in u]))
  norm_v = np.sqrt(np.sum([v**2 for v in v]))
  cosine_similarity = dot / (norm_u * norm_v)
  return cosine_similarity

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
  """
  Performs the word analogy task: a is to b as c is to ____.
  """
  word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

  e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

  words = word_to_vec_map.keys()
  max_cosine_sim = -100
  best_word = None

  for w in words:
    if w in [word_a, word_b, word_c]:
      continue

    cosine_sim = cosine_similarity((e_b-e_a), (word_to_vec_map[w] - e_c))

    if cosine_sim > max_cosine_sim:
      max_cosine_sim = cosine_sim
      best_word = w

  return best_word
