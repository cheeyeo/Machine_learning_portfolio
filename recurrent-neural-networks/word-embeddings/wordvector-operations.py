# Example on using GloVe embedding to perform word vector operations
# such as similarity comparision and analogy

import numpy as np
from wordvector_utils import read_glove_vecs, cosine_similarity, complete_analogy

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

father = word_to_vec_map['father']
mother = word_to_vec_map['mother']
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map['crocodile']
france = word_to_vec_map['france']
italy = word_to_vec_map['italy']
paris = word_to_vec_map['paris']
rome = word_to_vec_map['rome']

print("Cosine similarity (father, mother) = {}".format(cosine_similarity(father, mother)))
print("Cosine similarity (ball, crocodile) = {}".format(cosine_similarity(ball, crocodile)))
print("Cosine similarity (france - paris, rome - italy) = {}".format(cosine_similarity(france-paris, rome-italy)))

print()
print('Trying out analogy example\n')
tuples_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for t in tuples_to_try:
    print ('{} -> {} :: {} -> {}'.format(*t, complete_analogy(*t,word_to_vec_map)))
