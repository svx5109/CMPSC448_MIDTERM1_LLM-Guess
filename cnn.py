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

torch.set_num_threads(8)

# enable GPU if available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU enabled")
else:
    print("No GPU available")

# Vocab class
class Vocab:
    def __init__(self, max_input_length):
        paddingToken = "<PADDING>"
        unknownToken = "<UNKNOWN>"
        separatorToken = "<SEP>"

        self.max_length = max_input_length
        self.vocab = {
            paddingToken : 0, 
            unknownToken : 1,
            separatorToken : 2
            }
        
    # returns encoding tables of classes
    def create_class_ids(self, class_column):
        labels = class_column.dropna().unique()
        
        self.class_to_id = {
            label: i for i, label in enumerate(sorted(labels))
        }
        self.id_to_class = {
            i: label for label, i in self.class_to_id.items()
        }

    def encode_output_column(self, output_column):
        self.create_class_ids(output_column)
        return output_column.map(self.class_to_id)

    # turns input text column into encoded token ids
    def encode_input_column(self, text_column):
        encodings = []
        for text in text_column:
            tokens = str(text).lower().split()
            encoding = []
            
            for i in range(0, self.max_length):
                if i < len(tokens):
                    token = tokens[i]
                    encoding.append(self.vocab.get(token, self.vocab["<UNKNOWN>"]))
                else:
                    encoding.append(self.vocab["<PADDING>"])
            encodings.append(encoding)
        return encodings

    def encode_input_row(self, row, columns):
        encoding = []
        col_num = 0

        #input then output
        for column in columns:
            text = row[column]
            tokens = str(text).lower().split()
            # creates all the tokens for a column then gets vocab ids
            for token in tokens:
                if len(encoding) < self.max_length:
                    encoding.append(self.vocab.get(token, self.vocab["<UNKNOWN>"]))
                else:
                    break

            #encode separator
            if col_num < len(columns) and len(encoding) < self.max_length:
                encoding.append(self.vocab["<SEP>"])

        #fill the rest with padding
        while len(encoding) < self.max_length:
            encoding.append(self.vocab["<PADDING>"])

        return encoding


    def encode_multiple_input_columns(self, input_df):
        encodings = []
        for _, row in input_df.iterrows():
            encoding = self.encode_input_row(row, input_df.columns)
            encodings.append(encoding)

        return encodings

    # adds all the tokens in an input column to the vocabulary
    def add_vocab_tokens(self, text_column):
        # produce encodings for different tokens
        for text in text_column:
            tokens = str(text).lower().split()
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
    
# CNN text classifier model
class CNNTextClassifier:
    # Initialize the model
    def __init__(self, x_train: tensor, y_train: tensor, x_validate: tensor, y_validate: tensor, x_test: tensor, y_test: tensor, vocab: dict[str, int], parameters: dict[str, int | float]) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.x_test = x_test
        self.y_test = y_test
        self.vocab = vocab
        self.parameters = parameters
        self.create_data_loaders()
        self.create_model()

    # Creates the data loaders. 
    # We must do it in batches with the bigger dataset because otherwise the program will crash due to not enough memory
    def create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        #train dataset
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.parameters["batch_size"], shuffle=self.parameters["shuffle"])
        #validate dataset
        self.validate_dataset = TensorDataset(self.x_validate, self.y_validate)
        self.validate_loader = DataLoader(self.validate_dataset, batch_size=self.parameters["batch_size"], shuffle=self.parameters["shuffle"])
        return self.train_loader, self.validate_loader

    # Creates the 1D CNN model.
    def create_model(self) -> None:
        self.embedding = nn.Embedding(len(self.vocab), self.parameters["embedding_dimension"], self.vocab["<PADDING>"]).to(device)
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels = self.parameters["embedding_dimension"], out_channels = self.parameters["out_channels"], kernel_size = self.parameters["kernel_size"]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(self.parameters["dropout"]),
            nn.Linear(self.parameters["out_channels"], self.parameters["classes"])
        ).to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(list(self.embedding.parameters()) + list(self.classifier.parameters()), lr=1e-3)

    # Gets the scores from input data. Use softmax to determine predictions of classes.
    def get_scores(self, token_ids):
        x = self.embedding(token_ids)   # (batch, 512) -> (batch, 512, 32)
        x = x.transpose(1, 2)           # (batch, 512, 32) -> (batch, 32, 512)
        return self.classifier(x)       # (batch, 3)

    # Fits the model to the training data.
    def fit(self, print_loss: bool) -> None:
        if print_loss:
            print("fitting")
        
        epochs_no_improve = 0
    
        best_embedding_state = None
        best_classifier_state = None
        best_epoch = -1
        best_loss = float("inf")
    
        for epoch in range(self.parameters["max_epochs"]):        
            self.embedding.train()
            self.classifier.train()
    
            total_train_loss = 0.0
            total_train_items = 0
            
            total_validate_loss = 0.0
            total_validate_items = 0
    
            # gradient descent and training loss calculation
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
    
                self.optimizer.zero_grad()
                scores = self.get_scores(batch_x)
                loss = self.loss_func(scores, batch_y)
                loss.backward()
                self.optimizer.step()
                
                batch_size = batch_x.size(0)
                total_train_loss += loss.item() * batch_size
                total_train_items += batch_size
    
            self.embedding.eval()
            self.classifier.eval()
    
            # validate loss calculation
            with torch.no_grad():
                for batch_x, batch_y in self.validate_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    scores = self.get_scores(batch_x)
                    loss = self.loss_func(scores, batch_y)
                    
                    batch_size = batch_x.size(0)
                    total_validate_loss += loss.item() * batch_size
                    total_validate_items += batch_size
    
            # epoch loss calculation
            train_loss = total_train_loss / total_train_items
            validate_loss = total_validate_loss / total_validate_items

            if print_loss:
                print("Epoch:", str(epoch), "Train Loss:" , str(train_loss), "Validate Loss:", str(validate_loss))
    
            # early stopping
    
            # calculate if improved
            if validate_loss < best_loss:
                best_loss = validate_loss
                best_embedding_state = {
                    key: value.detach().clone() for key, value in self.embedding.state_dict().items()
                }
                best_classifier_state = {
                    key: value.detach().clone() for key, value in self.classifier.state_dict().items()
                }
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
    
            # if more epochs without improvement, stop training
            if epochs_no_improve >= self.parameters["early_stopping_epochs"]:
                break
    
        #restore best state
        self.embedding.load_state_dict(best_embedding_state)
        self.classifier.load_state_dict(best_classifier_state)
        
    # Tests the model against its test sets and prints accuracy and loss.
    def print_evaluation(self) -> None:
        with torch.no_grad():    
            scores = self.get_scores(self.x_test.to(device))
            test_loss = self.loss_func(scores, self.y_test.to(device))
            test_predictions = torch.argmax(scores, dim=1)
            test_probabilities = torch.softmax(scores, dim=1)

            print("Loss:", test_loss.item())
            print("Accuracy:", accuracy_score(self.y_test, test_predictions.cpu()))


# load the data
dataDF = pandas.read_csv("train_v2_drcat_02.csv")

llm_label = "source"
input_label = "prompt_name"
output_label = "text"

vocab = Vocab(512)

#transform data into a suitable form
xDF = dataDF[output_label]
yDF = dataDF[llm_label]
yDF = vocab.encode_output_column(yDF)

xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, test_size = 0.2, stratify = yDF, random_state = 67)
xTrain, xValidate, yTrain, yValidate = train_test_split(xTrain, yTrain, test_size= 0.2 / 0.8, stratify = yTrain, random_state=41)

#create tensor vectors
vocab.add_vocab_tokens(xTrain)

xTrain = tensor(vocab.encode_input_column(xTrain), dtype = torch.long)
xValidate = tensor(vocab.encode_input_column(xValidate), dtype = torch.long)
xTest = tensor(vocab.encode_input_column(xTest), dtype = torch.long)
yTrain = tensor(yTrain.to_numpy(), dtype = torch.long)
yValidate = tensor(yValidate.to_numpy(), dtype = torch.long)
yTest = tensor(yTest.to_numpy(), dtype = torch.long)

print(xTrain.shape)
print(yTrain.shape)

parameters = {
    "embedding_dimension" : 150,
    "out_channels" : 16,
    "kernel_size" : 5,
    "dropout" : 0.2,
    "early_stopping_epochs" : 5,
    "max_epochs" : 1000,
    "batch_size" : 32,
    "shuffle" : 2,
    "classes" : len(vocab.class_to_id)
    }


cnn_model = CNNTextClassifier(xTrain, yTrain, xValidate, yValidate, xTest, yTest, vocab.vocab, parameters)
cnn_model.fit(True)
cnn_model.print_evaluation()
# Accuracy: 0.9134, Loss: 0.2274

# try CNN with both input and output
xDF = dataDF[[input_label, output_label]]
xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, test_size = 0.2, stratify = yDF, random_state = 67)
xTrain, xValidate, yTrain, yValidate = train_test_split(xTrain, yTrain, test_size= 0.2 / 0.8, stratify = yTrain, random_state=41)

vocab_2 = Vocab(512)
vocab_2.add_vocab_tokens(xTrain[input_label])
vocab_2.add_vocab_tokens(xTrain[output_label])

xTrain = tensor(vocab_2.encode_multiple_input_columns(xTrain), dtype = torch.long)
xValidate = tensor(vocab_2.encode_multiple_input_columns(xValidate), dtype = torch.long)
xTest = tensor(vocab_2.encode_multiple_input_columns(xTest), dtype = torch.long)
yTrain = tensor(yTrain.to_numpy(), dtype = torch.long)
yValidate = tensor(yValidate.to_numpy(), dtype = torch.long)
yTest = tensor(yTest.to_numpy(), dtype = torch.long)

cnn_model_2 = CNNTextClassifier(xTrain, yTrain, xValidate, yValidate, xTest, yTest, vocab_2.vocab, parameters)
cnn_model_2.fit(True)
cnn_model_2.print_evaluation()
# Accuracy: 0.9185, Loss: 0.2131
# This approach increases accuracy by a marginal amount in exchange for a larger embedding and thus slower training