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

# enable GPU if available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU enabled")
else:
    print("No GPU available")

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
xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, test_size = 0.2, stratify = yDF, random_state = 67)
xTrain, xValidate, yTrain, yValidate = train_test_split(xTrain, yTrain, test_size= 0.2 / 0.8, stratify = yTrain, random_state=41)

# TODO: Replace this with an actual good tokenizer libary (ie tiktoken)
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
xValidate = torch.tensor(encode(xValidate), dtype = torch.long)
xTest = torch.tensor(encode(xTest), dtype = torch.long)
yTrain = torch.tensor(yTrain.to_numpy(), dtype = torch.long)
yValidate = torch.tensor(yValidate.to_numpy(), dtype = torch.long)
yTest = torch.tensor(yTest.to_numpy(), dtype = torch.long)

print(xTrain.shape)
print(yTrain.shape)

# we must do it in batches with the bigger dataset because otherwise the program will crash due to not enough memory
train_dataset = TensorDataset(xTrain, yTrain)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=2)

validate_dataset = TensorDataset(xValidate, yValidate)
validate_loader = DataLoader(validate_dataset, batch_size=32, shuffle=2)

#create the model
parameters = {
    "embedding_dimension" : 100,
    "out_channels" : 16,
    "kernel_size" : 5,
    "dropout" : 0.1,
    "early_stopping_epochs" : 10,
    "max_epochs" : 1000,
    }


embedding = nn.Embedding(len(vocab), parameters["embedding_dimension"], vocab[paddingToken]).to(device)
classifier = nn.Sequential(
    nn.Conv1d(in_channels = parameters["embedding_dimension"], out_channels = parameters["out_channels"], kernel_size = parameters["kernel_size"]),
    nn.ReLU(),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten(),
    nn.Dropout(parameters["dropout"]),
    nn.Linear(parameters["out_channels"], len(class_encodings))
).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(embedding.parameters()) + list(classifier.parameters()), lr=1e-3)

# train the model
def get_scores(token_ids):
    x = embedding(token_ids)  # (batch, 512) -> (batch, 512, 32)
    x = x.transpose(1, 2)    # (batch, 512, 32) -> (batch, 32, 512)
    return classifier(x)     # (batch, 3)

def fit_model():
    print("fitting")

    epochs_no_improve = 0

    best_embedding_state = None
    best_classifier_state = None
    best_epoch = -1
    best_loss = float("inf")


    for epoch in range(parameters["max_epochs"]):        
        embedding.train()
        classifier.train()

        total_train_loss = 0.0
        total_train_items = 0
        
        total_validate_loss = 0.0
        total_validate_items = 0

        # gradient descent and training loss calculation
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            scores = get_scores(batch_x)
            loss = loss_func(scores, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_size = batch_x.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_items += batch_size

        embedding.eval()
        classifier.eval()

        # validate loss calculation
        with torch.no_grad():
            for batch_x, batch_y in validate_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                scores = get_scores(batch_x)
                loss = loss_func(scores, batch_y)
                
                batch_size = batch_x.size(0)
                total_validate_loss += loss.item() * batch_size
                total_validate_items += batch_size

        # epoch loss calculation
        train_loss = total_train_loss / total_train_items
        validate_loss = total_validate_loss / total_validate_items
        
        print("Epoch:", str(epoch), " , Train Loss:" , str(train_loss), " Validate Loss: ", str(validate_loss))

        # early stopping

        # calculate if improved
        if validate_loss < best_loss:
            best_loss = validate_loss
            best_embedding_state = {
                key: value.detach().clone() for key, value in embedding.state_dict().items()
            }
            best_classifier_state = {
                key: value.detach().clone() for key, value in classifier.state_dict().items()
            }
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # if more epochs without improvement, stop training
        if epochs_no_improve >= parameters["early_stopping_epochs"]:
            break

    #restore best parameters
    embedding.load_state_dict(best_embedding_state)
    classifier.load_state_dict(best_classifier_state)


torch.set_num_threads(8)
fit_model()

# evaluate the model
embedding.eval()
classifier.eval()

with torch.no_grad():    
    scores = get_scores(xTest.to(device))
    testLoss = loss_func(scores, yTest.to(device))
    testPredictions = torch.argmax(scores, dim=1)
    testProbabilities = torch.softmax(scores, dim=1)

    print("Loss:", testLoss.item())
    print("Accuracy:", accuracy_score(yTest, testPredictions.cpu()))

#latest accuracy: 0.807443

#print("Predictions:")
#print(testPredictions)
#print("Correct:")
#print(yTest)
#print(loss_func(testPredictions, yTest))
