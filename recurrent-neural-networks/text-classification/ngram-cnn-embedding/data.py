from utils import load_clean_dataset, save_dataset, load_dataset
import numpy as np

trainDocs, ytrain = load_clean_dataset(True)
testDocs, ytest = load_clean_dataset(False)

save_dataset([trainDocs, ytrain], "data/train.pkl")
save_dataset([testDocs, ytest], "data/test.pkl")

# (dataset,labels) = load_dataset("data/train.pkl")
# print(np.array(dataset).shape)
