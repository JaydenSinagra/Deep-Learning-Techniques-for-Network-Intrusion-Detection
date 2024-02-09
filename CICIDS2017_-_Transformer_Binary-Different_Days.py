#!/usr/bin/env python
# coding: utf-8

import time
import csv

import numpy as np
import pandas as pd

# Reference - https://github.com/fisher85/ml-cybersecurity/blob/master/python-web-attack-detection/web-attack-detection.ipynb

file_1 = pd.read_csv('balanced_multiclass_1.csv')

# Replacing non-numerical values
file_1.replace('Infinity', -1, inplace = True)

# Replacing NaN and Infinity values with -1
file_1.replace([np.inf, -np.inf, np.nan], -1, inplace = True)

# Converting 'Attack_Label' into binary format
file_1['Attack_Label'] = np.where(file_1['Attack_Label'] == 'BENIGN', 0, 1)

# Extracting 'Attack_Label' as 'y'
y = file_1['Attack_Label'].values

# Extracting remaining columns as 'x'
x = file_1.drop(columns = ['Source_IP', 'Destination_IP', 'Timestamp', 'Attack_Label'])

print("Before Scaling:")
print("\nY Values:")
print(y)
print("\nX Values:")
print(x)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Fitting and transforming the features using Min-Max Scaling
x = scaler.fit_transform(x)

print("After Scaling:")
print("\nY Values:")
print(y)
print("\nX Values:")
print(x)

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, test_size = 0.3, random_state = 42)

print("Y Train:")
print(y_train)
print("\nX Train:")
print(x_train)

print("\nY Test:")
print(y_test)
print("\nX Test:")
print(x_test)

# Converting to Pandas Series
y_train_series = pd.Series(y_train)
y_test_series = pd.Series(y_test)

print("\nY Train value counts:")
print(y_train_series.value_counts())
print("\nY Test value counts:")
print(y_test_series.value_counts())

file_2 = pd.read_csv('balanced_multiclass_2.csv')

# Replacing non-numerical values
file_2.replace('Infinity', -1, inplace = True)

# Replacing NaN and Infinity values with -1
file_2.replace([np.inf, -np.inf, np.nan], -1, inplace = True)

# Converting 'Attack_Label' into binary format
file_2['Attack_Label'] = np.where(file_2['Attack_Label'] == 'BENIGN', 0, 1)

# Extracting 'Attack_Label' as 'y'
y = file_2['Attack_Label'].values

# Extracting remaining columns as 'x'
x = file_2.drop(columns = ['Source_IP', 'Destination_IP', 'Timestamp', 'Attack_Label'])

print("Before Scaling:")
print("\nY Values:")
print(y)
print("\nX Values:")
print(x)

scaler = MinMaxScaler()

# Fitting and transforming the features using Min-Max Scaling
x = scaler.fit_transform(x)

print("After Scaling:")
print("\nY Values:")
print(y)
print("\nX Values:")
print(x)

# Concatenating x_test with x
x_test = np.concatenate((x_test, x), axis = 0)

# Concatenating y_test with y
y_test = np.concatenate((y_test, y), axis = 0)

print("\nX Train:")
print(x_train)
print("\nY Train:")
print(y_train)
print("\nX Test:")
print(x_test)
print("\nY Test:")
print(y_test)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# Transformer Network

# Reference [2] : A. Sarkar, “A comprehensive guide to building a transformer model with pytorch”, DataCamp. [Online]. Available : 
# https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch. 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensuring that the model dimension (d_model) is divisible by the number of heads (num_heads)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initializing dimensions
        self.d_model = d_model # Model's dimension (d_model)
        self.num_heads = num_heads # Number of attention heads (num_heads)
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value (d_k)

        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation (W_q)
        self.W_k = nn.Linear(d_model, d_model) # Key transformation (W_k)
        self.W_v = nn.Linear(d_model, d_model) # Value transformation (W_v)
        self.W_o = nn.Linear(d_model, d_model) # Output transformation (W_o)

    def scaled_dot_product_attention(self, Q, K, V, mask = None):
        # Calculating attention scores (attn_scores)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Applying mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Applying Softmax to obtain attention probabilities (attn_probs)
        attn_probs = torch.softmax(attn_scores, dim = -1)

        # Multiplying by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        # Reshaping the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # Combining the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask = None):
        # Applying linear transformations and splitting heads
        Q = self.split_heads(self.W_q(Q)) # Query (Q)
        K = self.split_heads(self.W_k(K)) # Key (K)
        V = self.split_heads(self.W_v(V)) # Value (V)

        # Performing scaled dot-product attention (attn_output)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combining heads and applying output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Reference [2]

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Reference [2]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        # Calculating positional encodings (pe)
        pe = self.positional_encoding(max_seq_length, d_model)
        self.register_buffer('pe', pe)

    def positional_encoding(self, max_seq_length, d_model):
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Calculating positional encodings (pe)
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Reference [2]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Reference [2]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
# Reference [2]

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.transformer_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, x):
        src_mask = self.generate_mask(x)
        x = self.embedding(x)
        x = self.dropout(self.positional_encoding(x))
        for layer in self.transformer_layers:
            x = layer(x, src_mask)

        # Global max pooling over the sequence length
        x = torch.max(x, dim = 1)[0]
        x = self.fc(x)
        return x

# Defining parameters for the model
max_seq_length = x_train.shape[1]
input_dim = int(np.max(x_train) + 1)
num_classes = len(np.unique(y_train))
d_model = 8
num_heads = 1
num_layers = 1
d_ff = 32
dropout = 0.2
num_epochs = 10
batch_size = 8

all_labels = np.concatenate((y_train, y_test))

# Encoding labels using LabelEncoder
label_encoder = LabelEncoder()

all_labels_encoded = label_encoder.fit_transform(all_labels)
y_train_encoded = all_labels_encoded[:len(y_train)]
y_test_encoded = all_labels_encoded[len(y_train):]

# Creating PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype = torch.long)
y_train_tensor = torch.tensor(y_train_encoded, dtype = torch.long)

x_test_tensor = torch.tensor(x_test, dtype = torch.long)
y_test_tensor = torch.tensor(y_test_encoded, dtype = torch.long)

transformer = TransformerClassifier(input_dim, num_classes, d_model, num_heads, 
                                    num_layers, d_ff, max_seq_length, dropout)

# Assigning the device based on GPU availability, defaults to CPU if no GPU is found
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.to(device)

# Defining loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer.parameters(), lr = 0.001, betas = (0.9, 0.98), eps = 1e-9)

train_dataset = data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_dataset = data.TensorDataset(x_test_tensor, y_test_tensor)
test_loader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

start = time.time()

# Training the model
for epoch in range(num_epochs):
    transformer.train()
    total_loss = 0

    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = transformer(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    
    print("Epoch [{}/{}] - Loss: {:.4f}".format(epoch + 1, num_epochs, average_loss))

end = time.time()

print("Time consumed to fit model is ", end - start)

# Evaluating the model 
transformer.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        outputs = transformer(batch_x)
        
        predictions.extend(outputs.argmax(dim = 1).cpu().numpy())
        true_labels.extend(batch_y.cpu().numpy())

print(classification_report(true_labels, predictions, zero_division = 0))

# Generating the Confusion Matrix
confusion_mat = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:")
print(confusion_mat)
