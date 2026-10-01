# this is the 1D CNN neural network to classify text into 3 different LLMs: GPT, Claude, and Gemini

# import libraries
print("loading libraries")
# import sklearn
import pandas
import torch
# import re
from sklearn.model_selection import train_test_split

# load the data
print("loading data")
dataDF = pandas.read_csv("C:/Users/xiest/OneDrive - The Pennsylvania State University/Documents/CMPSC 448/Homework/MIDTERM1_LLM-Guess/dataset_small.csv")
#dataDF.info()

#transform data into a suitable form
MAX_LENGTH = 512
encoding = {
    "GPT-5.6" : 0,
    "Claude 4.5" : 1,
    "Gemini 3.6" : 2,
}
dataDF["llm"] = dataDF["llm"].map(encoding)

xDF = dataDF["output"]
yDF = dataDF["llm"]

xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, test_size = 0.3, stratify = yDF, random_state = 67)

# tokenize
paddingToken = "<PADDING>"
unknownToken = "<UNKNOWN>"
vocab = {
    paddingToken : 0, 
    unknownToken : 1
    }

# produce encodings for different tokens
for text in xTrain:
    tokens = str(text).lower().split()

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

# turns output data into columns
def encode(text_column):
    encodings = []
    for text in text_column:
        tokens = str(text).lower().split()
        encoding = []
        
        for i in range(0, MAX_LENGTH):
            if i < len(tokens):
                token = tokens[i]
                encoding.append(vocab.get(token, vocab[unknownToken]))
            else:
                encoding.append(vocab[paddingToken])
        encodings.append(encoding)
    return encodings


xTrain = torch.tensor(encode(xTrain), dtype = torch.long)
xTest = torch.tensor(encode(xTest), dtype = torch.long)

print(xTrain[0])
print(xTrain.shape)

