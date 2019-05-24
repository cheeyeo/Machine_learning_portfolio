from utils import *
import numpy as np

doc = load_doc('deu.txt')

pairs = to_pairs(doc)

clean_pairs = clean_pairs(pairs)
for i in range(100):
  print('[{:s}] => [{:s}]'.format(clean_pairs[i,0], clean_pairs[i,1]))

save_clean_data(clean_pairs, 'english-german.pkl')

# Load dataset
data = load_clean_sentences('english-german.pkl')

# Reduce the dataset to first 10000 examples
# Take 9000 for training and 1000 for testing
samples = 10000
dataset = data[:samples, :]
np.random.shuffle(dataset)

train, test = dataset[:9000], dataset[9000:]

# Save the datasets
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')
