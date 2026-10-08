# this is the 1D CNN neural network to classify text into 3 different LLMs: GPT, Claude, and Gemini

# import libraries
print("loading libraries")
import pandas
import torch
from torch import nn
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import numpy

# load the data
dataDF = pandas.read_csv("train_v2_drcat_02.csv")

llm_label = "source"
input_label = "prompt_name"
output_label = "text"

# create class encodings
labels = dataDF[llm_label].dropna().unique()
class_encodings = {
    label: i for i, label in enumerate(sorted(labels))
}
id_to_class = {
    i: label for label, i in class_encodings.items()
}
dataDF[llm_label] = dataDF[llm_label].map(class_encodings)

#transform data into a suitable form
xDF = dataDF[output_label]
yDF = dataDF[llm_label]
xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, test_size = 0.3, stratify = yDF, random_state = 67)


# Create the vocabulary
MAX_LENGTH = 512

paddingToken = "<PADDING>"
unknownToken = "<UNKNOWN>"
vocab = {
    paddingToken : 0, 
    unknownToken : 1
    }

# adds tokens into vocabulary
def add_vocab(text_column):
    # produce encodings for different tokens
    for text in text_column:
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
add_vocab(xTrain)

xTrain = torch.tensor(encode(xTrain), dtype = torch.long)
xTest = torch.tensor(encode(xTest), dtype = torch.long)
yTrain = torch.tensor(yTrain.to_numpy(), dtype = torch.long)
yTest = torch.tensor(yTest.to_numpy(), dtype = torch.long)

print(xTrain.shape)
print(yTrain.shape)

# we must do it in batches with the bigger dataset because otherwise the program will crash due to not enough memory
train_dataset = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=2)

#create the model
parameters = {
    "embedding_dimension" : 100,
    "out_channels" : 8,
    "kernel_size" : 3,
    "dropout" : 0.5
    }


embedding = nn.Embedding(len(vocab), parameters["embedding_dimension"], vocab[paddingToken])
classifier = nn.Sequential(
    nn.Conv1d(in_channels = parameters["embedding_dimension"], out_channels = parameters["out_channels"], kernel_size = parameters["kernel_size"]),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Dropout(parameters["dropout"]),
    nn.Linear(parameters["out_channels"], len(class_encodings))
)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(embedding.parameters()) + list(classifier.parameters()), lr=1e-3)

# train the model
def get_scores(token_ids):
    x = embedding(token_ids)  # (batch, 512) -> (batch, 512, 32)
    x = x.transpose(1, 2)    # (batch, 512, 32) -> (batch, 32, 512)
    return classifier(x)     # (batch, 3)

def fit_model(xTrain, epochs):
    print("fitting")

    for epoch in range(epochs):
        total_loss = 0.0
        embedding.train()
        classifier.train()
        
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            scores = get_scores(batch_x)
            loss = loss_func(scores, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print("Epoch:", str(epoch), " , Loss:" , str(total_loss), ", Accuracy:")
        #print("Epoch:", str(epoch), " , Loss:" , str(loss.item()), ", Accuracy:", accuracy_score(yTrain, torch.argmax(scores, dim=1)))


fit_model(xTrain, 15)

# evaluate the model
embedding.eval()
classifier.eval()

with torch.no_grad():
    testPredictions = torch.argmax(get_scores(xTest), dim=1)
    testProbabilities = torch.softmax(get_scores(xTest), dim=1)
    print("Accuracy: ", accuracy_score(yTest, testPredictions))


#print("Predictions:")
#print(testPredictions)
#print("Correct:")
#print(yTest)
#print(loss_func(testPredictions, yTest))
