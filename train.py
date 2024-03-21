import json
from nltk_util import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet


with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []


# Step 1: Create List of all Words
# Also stores each sentence in patterns along with corresponding tag in 'xy'
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# List of punctuations to be ignored
ignore_words = ['?', '!', ',', '.', ';']


# Removes punctuations from all_words
temp = []
for word in all_words:
    if word not in ignore_words:
        temp.append(stem(word))
all_words = np.array(temp)    

# Sorts tags and all_words in ascending order
all_words = sorted(np.unique(all_words))
tags = sorted(np.unique(tags))


# Create training data
x_train = []    # Stores the sentence vector (1-hot encoded, sort of)
y_train = []    # Stores the index of the corresponding tag.
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)     # Tags is a list. This line finds the index of `tag` in that list.
    y_train.append(label)       

x_train = np.array(x_train)
y_train = np.array(y_train)

# print(x_train)


# Hyper Parameters
batch_size = 8
hidden_size = 8     # Number of nodes in hidden layer
input_size = len(x_train[0])
output_size = len(tags)

learning_rate = 0.001
num_epochs = 1000


# This class is used to transform the training data into data that can be used as Pytorch NN input
class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    # Allows us to access a dataset with an index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
    
    def __str__(self):
        string = f'{self.x_data}\n{self.y_data}'
        return string


# Takes training data and transforms it into NN input
dataset = ChatDataset(x_train, y_train)
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, 
                          shuffle = True, num_workers = 0)

# Checks if GPU available, else uses CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()   # Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)    # Optimization Function


# Optimization Loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and Optimizer Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch = {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}')

print(f'Final Loss = {loss.item():.4f}')


# Here we will store the trained model into a file,
# so that we won't have to train the model everytime we use the bot.
# Data that will be stored in this dictionary format
data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags
}

FILE = 'data.pth'
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')