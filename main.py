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
        inputToken = "<INPUT>"
        outputToken = "<OUTPUT>"

        self.max_length = max_input_length
        self.vocab = {
            paddingToken : 0, 
            unknownToken : 1,
            inputToken : 2,
            outputToken : 3
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

        for col_num, column in enumerate(columns):
            if col_num == 0 and len(encoding) < self.max_length:
                encoding.append(self.vocab["<INPUT>"])
            elif col_num == 1 and len(encoding) < self.max_length:
                encoding.append(self.vocab["<OUTPUT>"])

            text = row[column]
            tokens = str(text).lower().split()

            # creates all the tokens for a column then gets vocab ids
            for token in tokens:
                if len(encoding) < self.max_length:
                    encoding.append(self.vocab.get(token, self.vocab["<UNKNOWN>"]))
                else:
                    break

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

    """
    nn.Conv1d(in_channels = self.parameters["out_channels"], out_channels = self.parameters["out_channels"], kernel_size = self.parameters["kernel_size"]),
    nn.ReLU(),
    nn.Conv1d(in_channels = self.parameters["out_channels"], out_channels = self.parameters["out_channels"], kernel_size = self.parameters["kernel_size"]),
    nn.ReLU(),
    """ 

    # Creates the 1D CNN model.
    def create_model(self) -> None:
        self.embedding = nn.Embedding(len(self.vocab.vocab), self.parameters["embedding_dimension"], self.vocab.vocab["<PADDING>"]).to(device)
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels = self.parameters["embedding_dimension"], out_channels = self.parameters["out_channels"], kernel_size = self.parameters["kernel_size"]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(self.parameters["dropout"]),
            nn.Linear(self.parameters["out_channels"], len(self.vocab.class_to_id))
        ).to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(self.embedding.parameters()) + 
            list(self.classifier.parameters()), 
            lr=self.parameters["learning_rate"])

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
            report = classification_report(
                self.y_test.numpy(),
                test_predictions.cpu(),
                labels=list(range(len(self.vocab.id_to_class))),
                target_names=[self.vocab.id_to_class[i] for i in range(len(self.vocab.id_to_class))],
                zero_division=0)

            print("Loss:", test_loss.item())
            print("Accuracy:", accuracy_score(self.y_test, test_predictions.cpu()))
            print("Report:\n", report)
        
# RNN text classifier model
class RNNTextClassifier:
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
    def create_data_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        #train dataset
        self.train_dataset = TensorDataset(self.x_train, self.y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.parameters["batch_size"], shuffle=self.parameters["shuffle"])
        #validate dataset
        self.validate_dataset = TensorDataset(self.x_validate, self.y_validate)
        self.validate_loader = DataLoader(self.validate_dataset, batch_size=self.parameters["batch_size"], shuffle=self.parameters["shuffle"])
        #test dataset (because the testing is too big)
        self.test_dataset = TensorDataset(self.x_test, self.y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.parameters["batch_size"], shuffle=self.parameters["shuffle"])
        return self.train_loader, self.validate_loader, self.test_loader

    # Creates the LSTM RNN model
    def create_model(self) -> None:
        self.embedding = nn.Embedding(len(self.vocab.vocab), self.parameters["embedding_dimension"], self.vocab.vocab["<PADDING>"]).to(device)
        self.lstm = nn.LSTM(
            input_size=self.parameters["embedding_dimension"],
            hidden_size=self.parameters["hidden_size"],
            num_layers=self.parameters["num_layers"],
            batch_first=True,
            dropout=(self.parameters["dropout"])
        ).to(device)
        self.dropout = nn.Dropout(self.parameters["dropout"]).to(device)
        self.classifier = nn.Linear(self.parameters["hidden_size"], len(self.vocab.class_to_id)).to(device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(self.embedding.parameters()) + 
            list(self.lstm.parameters()) +
            list(self.classifier.parameters()),    
            lr=self.parameters["learning_rate"])

    def get_scores(self, token_ids):
        x = self.embedding(token_ids)
        _, (hidden, _) = self.lstm(x)
        sequence_vec = hidden[-1]
        return self.classifier(self.dropout(sequence_vec))

    # Fits the model to the training set.
    def fit(self, print_loss: bool) -> None:
        best_loss = float("inf")
        best_embedding_state = None
        best_lstm_state = None
        best_classifier_state = None
        best_epoch = -1

        if print_loss:
            print("fitting RNN")

        for epoch in range(self.parameters["max_epochs"]):
            self.embedding.train()
            self.lstm.train()
            self.dropout.train()
            self.classifier.train()

            total_train_loss = 0.0
            total_train_items = 0
            
            total_validate_loss = 0.0
            total_validate_items = 0

            # gradient descent and training loss
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
            self.lstm.eval()
            self.dropout.eval()
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

            # calculate if improved
            if validate_loss < best_loss:
                best_loss = validate_loss
                best_embedding_state = {
                    key: value.detach().clone() for key, value in self.embedding.state_dict().items()
                }
                best_lstm_state = {
                    key: value.detach().clone() for key, value in self.lstm.state_dict().items()
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
        self.lstm.load_state_dict(best_lstm_state)
        self.classifier.load_state_dict(best_classifier_state)

    def print_evaluation(self) -> None:
        self.embedding.eval()
        self.lstm.eval()
        self.dropout.eval()
        self.classifier.eval()
        
        with torch.no_grad():    
            total_test_items = 0
            total_test_loss = 0.0
            predictions = []
            targets = []
            
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                scores = self.get_scores(batch_x)
                loss = self.loss_func(scores, batch_y)
                batch_size = batch_x.size(0)
                total_test_loss += loss.item() * batch_size
                total_test_items += batch_size

                test_predictions = torch.argmax(scores, dim=1)
                predictions.extend(test_predictions.cpu().tolist())
                targets.extend(batch_y.cpu().tolist())
                            
            #test_probabilities = torch.softmax(scores, dim=1)
            report = classification_report(
                targets,
                predictions,
                labels=list(range(len(self.vocab.id_to_class))),
                target_names=[self.vocab.id_to_class[i] for i in range(len(self.vocab.id_to_class))],
                zero_division=0)

            print("Accuracy:", accuracy_score(targets, predictions))
            print("Loss:", total_test_loss/total_test_items)
            print("Report:\n", report)

RANDOM_SAMPLE = False
SAMPLE_SIZE = 40000
GROUP_BY_FAMILY = False
LIMIT_LLMS = True

llm_label = "source"
input_label = "prompt_name"
output_label = "text"
dataDF = None

# load/transform dataset

# lmsys chatbot arena
dataDF = pandas.read_parquet("hf://datasets/lmsys/chatbot_arena_conversations/data/train-00000-of-00001-cced8514c7ed782a.parquet")

rows = []

# collect all llm inputs and outputs, two on each side
for _, row in dataDF.iterrows():
    # Model A
    conversation = row["conversation_a"]
    if len(conversation) >= 1:
        user_input = conversation[0]
        llm_output = conversation[1]
        if user_input["role"] == "user" and llm_output["role"] == "assistant":
            rows.append({
                llm_label: row["model_a"],
                input_label: user_input["content"],
                output_label: llm_output["content"]
            })

    # Model B
    conversation = row["conversation_b"]
    if len(conversation) >= 1:
        user_input = conversation[0]
        llm_output = conversation[1]
        if user_input["role"] == "user" and llm_output["role"] == "assistant":
            rows.append({
                llm_label: row["model_b"],
                input_label: user_input["content"],
                output_label: llm_output["content"]
            })


dataDF = pandas.DataFrame(rows)

if RANDOM_SAMPLE:
    dataDF = dataDF.sample(n=SAMPLE_SIZE, random_state=67).reset_index(drop=False)

if GROUP_BY_FAMILY:
    model_to_family = {
        "gpt-3.5-turbo": "GPT",
        "gpt-4": "GPT",

        "claude-instant-v1": "Claude",
        "claude-v1": "Claude",

        "alpaca-13b": "LLaMA-family",
        "gpt4all-13b-snoozy": "LLaMA-family",
        "guanaco-33b": "LLaMA-family",
        "koala-13b": "LLaMA-family",
        "llama-13b": "LLaMA-family",
        "vicuna-13b": "LLaMA-family",
        "vicuna-7b": "LLaMA-family",
        "wizardlm-13b": "LLaMA-family",

        "dolly-v2-12b": "Pythia-family",
        "oasst-pythia-12b": "Pythia-family",
    }
    dataDF[llm_label] = dataDF[llm_label].replace(model_to_family)

    if LIMIT_LLMS:
        families = {"Claude", "GPT", "LLaMA-family"}
        dataDF = dataDF[dataDF[llm_label].isin(families)]
else:
    if LIMIT_LLMS:
        llms = {"gpt-4", "claude-v1", "palm-2"}
        dataDF = dataDF[dataDF[llm_label].isin(llms)]


vocab = Vocab(256)

#transform data into a suitable form
xDF = dataDF[output_label]
yDF = dataDF[llm_label]
yDF = vocab.encode_output_column(yDF)


# Ouput only, CNN
xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, test_size = 0.2, stratify = yDF, random_state = 67)
xTrain, xValidate, yTrain, yValidate = train_test_split(xTrain, yTrain, test_size= 0.2 / 0.8, stratify = yTrain, random_state=41)

# add vocab
vocab.add_vocab_tokens(xTrain)

# create tensor vectors
xTrain = tensor(vocab.encode_input_column(xTrain), dtype = torch.long)
xValidate = tensor(vocab.encode_input_column(xValidate), dtype = torch.long)
xTest = tensor(vocab.encode_input_column(xTest), dtype = torch.long)
yTrain = tensor(yTrain.to_numpy(), dtype = torch.long)
yValidate = tensor(yValidate.to_numpy(), dtype = torch.long)
yTest = tensor(yTest.to_numpy(), dtype = torch.long)

print(xTrain.shape)
print(xValidate.shape)
print(xTest.shape)

# Create CNN model
cnn_parameters = {
    "embedding_dimension" : 200,
    "out_channels" : 16,
    "kernel_size" : 10,
    "dropout" : 0.40,
    "early_stopping_epochs" : 30,
    "max_epochs" : 1000,
    "batch_size" : 128,
    "shuffle" : 2,
    "learning_rate" : 1e-4
    }


cnn_model = CNNTextClassifier(xTrain, yTrain, xValidate, yValidate, xTest, yTest, vocab, cnn_parameters)
cnn_model.fit(True)
cnn_model.print_evaluation()

# try CNN with input only
xDF2 = dataDF[input_label]
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(xDF2, yDF, test_size = 0.2, stratify = yDF, random_state = 67)
xTrain2, xValidate2, yTrain2, yValidate2 = train_test_split(xTrain2, yTrain2, test_size= 0.2 / 0.8, stratify = yTrain2, random_state=41)

vocab_2 = Vocab(128)
vocab_2.class_to_id = vocab.class_to_id
vocab_2.id_to_class = vocab.id_to_class
vocab_2.add_vocab_tokens(xTrain2)

xTrain2 = tensor(vocab_2.encode_input_column(xTrain2), dtype = torch.long)
xValidate2 = tensor(vocab_2.encode_input_column(xValidate2), dtype = torch.long)
xTest2 = tensor(vocab_2.encode_input_column(xTest2), dtype = torch.long)
yTrain2 = tensor(yTrain2.to_numpy(), dtype = torch.long)
yValidate2 = tensor(yValidate2.to_numpy(), dtype = torch.long)
yTest2 = tensor(yTest2.to_numpy(), dtype = torch.long)

cnn_model_2 = CNNTextClassifier(xTrain2, yTrain2, xValidate2, yValidate2, xTest2, yTest2, vocab_2, cnn_parameters)
cnn_model_2.fit(True)
cnn_model_2.print_evaluation()
# Accuracy: 0.5794, Loss: 1.4168

# try CNN with both input and output
xDF3 = dataDF[[input_label, output_label]]
xTrain3, xTest3, yTrain3, yTest3 = train_test_split(xDF3, yDF, test_size = 0.2, stratify = yDF, random_state = 67)
xTrain3, xValidate3, yTrain3, yValidate3 = train_test_split(xTrain3, yTrain3, test_size= 0.2 / 0.8, stratify = yTrain3, random_state=41)

vocab_3 = Vocab(384)
vocab_3.class_to_id = vocab.class_to_id
vocab_3.id_to_class = vocab.id_to_class
vocab_3.add_vocab_tokens(xTrain3[input_label])
vocab_3.add_vocab_tokens(xTrain3[output_label])

xTrain3 = tensor(vocab_3.encode_multiple_input_columns(xTrain3), dtype = torch.long)
xValidate3 = tensor(vocab_3.encode_multiple_input_columns(xValidate3), dtype = torch.long)
xTest3 = tensor(vocab_3.encode_multiple_input_columns(xTest3), dtype = torch.long)
yTrain3 = tensor(yTrain3.to_numpy(), dtype = torch.long)
yValidate3 = tensor(yValidate3.to_numpy(), dtype = torch.long)
yTest3 = tensor(yTest3.to_numpy(), dtype = torch.long)

cnn_model_3 = CNNTextClassifier(xTrain3, yTrain3, xValidate3, yValidate3, xTest3, yTest3, vocab_3, cnn_parameters)
cnn_model_3.fit(True)
cnn_model_3.print_evaluation()

#RNN using the same output-only dataset
rnn_parameters = {
    "embedding_dimension" : 200,
    "hidden_size": 128,
    "num_layers" : 3,
    "dropout" : 0.15,
    "early_stopping_epochs" : 40,
    "max_epochs" : 1000,
    "batch_size" : 64,
    "shuffle" : 2,
    "learning_rate" : 1e-3,
    }

rnn_model = RNNTextClassifier(xTrain, yTrain, xValidate, yValidate, xTest, yTest, vocab, rnn_parameters)
rnn_model.fit(True)
rnn_model.print_evaluation()

# RNN with input only
rnn_model_2 = RNNTextClassifier(xTrain2, yTrain2, xValidate2, yValidate2, xTest2, yTest2, vocab_2, rnn_parameters)
rnn_model_2.fit(True)
rnn_model_2.print_evaluation()

# RNN with both input and output
rnn_model_3 = RNNTextClassifier(xTrain3, yTrain3, xValidate3, yValidate3, xTest3, yTest3, vocab_3, rnn_parameters)
rnn_model_3.fit(True)
rnn_model_3.print_evaluation()
