from pickle import load
from keras.models import load_model
import numpy as np
from utils import sample, print_sample

# load the model
model = load_model('model-final.h5')
# model.summary()
# load the mapping
char_to_ix = load(open('char_to_ix.pkl', 'rb'))
print(char_to_ix)
ix_to_char = load(open('ix_to_char.pkl', 'rb'))

sampled_indices = sample(model, char_to_ix, seq_length=27, n_chars=50)
print(sampled_indices)
print_sample(sampled_indices, ix_to_char)
