from utils import load_dataset
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


trainX, trainY = load_dataset('data/train.pkl')
testX, testY = load_dataset('data/test.pkl')

tokenizer = load_dataset('tokenizer.pkl')

length = max([len(s.split()) for s in trainX])
vocab_size = len(tokenizer.word_index) + 1
print("Max document length: {:d}".format(length))
print("Vocab size: {:d}".format(vocab_size))

encoded = tokenizer.texts_to_sequences(trainX)
trainX = pad_sequences(encoded, maxlen=length, padding='post')

encoded2 = tokenizer.texts_to_sequences(testX)
testX = pad_sequences(encoded2, maxlen=length, padding='post')

model = load_model('model.h5')

_, acc = model.evaluate([trainX, trainX, trainX], trainY, verbose=0)
print("Train accuracy: {:.2f}".format(acc * 100))

_, acc = model.evaluate([testX, testX, testX], testY, verbose=0)
print("Test accuracy: {:.2f}".format(acc * 100))
