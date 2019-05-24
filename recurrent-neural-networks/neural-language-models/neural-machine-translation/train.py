from utils import *
from model import *

dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# Create english tokenizer
eng_tokenizer = create_tokenizer(dataset[:,0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:,0])
print('English vocab size: {}'.format(eng_vocab_size))
print('English max length: {}'.format(eng_length))

# Create german tokenizer
ger_tokenizer = create_tokenizer(dataset[:,1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German vocab size: {}'.format(ger_vocab_size))
print('German max length: {}'.format(ger_length))

# Prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

# Prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
checkpoint = define_checkpoint()
model.fit(trainX, trainY,
          epochs=30,
          batch_size=64,
          validation_data=(testX, testY),
          callbacks=[checkpoint],
          verbose=1)
