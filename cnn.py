# this is the 1D CNN neural network to classify text into 3 different LLMs: GPT, Claude, and Gemini

# import libraries
print("loading libraries")
import pandas
import torch
from torch import nn

from sklearn.model_selection import train_test_split
import numpy

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


#create tensor vectors
xTrain = torch.tensor(encode(xTrain), dtype = torch.long)
xTest = torch.tensor(encode(xTest), dtype = torch.long)

yTrain = torch.tensor(yTrain.to_numpy(), dtype = torch.long)
yTest = torch.tensor(yTest.to_numpy(), dtype = torch.long)


print(xTrain)
print(xTrain.shape)
print(yTrain)
print(yTrain.shape)


#create the model
embedding_dimension = 100
embedding = nn.Embedding(len(vocab), embedding_dimension, vocab[paddingToken])
#need to transpose embedding because of expected input order in convolution layer

classifier = nn.Sequential(
    nn.Conv1d(in_channels = embedding_dimension, out_channels = 8, kernel_size = 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(8, 3)
)

def model(token_ids):
    x = embedding(token_ids)  # (batch, 512) -> (batch, 512, 32)
    x = x.transpose(1, 2)    # (batch, 512, 32) -> (batch, 32, 512)
    return classifier(x)     # (batch, 3)

print("fitting")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(embedding.parameters()) + list(classifier.parameters()),
    lr=1e-3,
)

for epoch in range(500):
    embedding.train()
    classifier.train()
    logits = model(xTrain)
    loss = criterion(logits, yTrain)
    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


trainPredictions = model(xTrain)
print("predict")
print(trainPredictions)
print("correct")
print(yTrain)

trainPredictions = model(xTest)
print("predict")
print(trainPredictions)
print("correct")
print(yTest)
print(criterion(trainPredictions, yTest))
