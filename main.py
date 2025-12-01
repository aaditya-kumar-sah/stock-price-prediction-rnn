import numpy as np 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
import torch.optim as optim


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error


ticker = 'AAPL'
df = yf.download(ticker, '2020-01-01')
print(df.head())

scaler = StandardScaler()

df['Close'] = scaler.fit_transform(df["Close"])

seq_len = 30
data = []

for i in range(len(df) - seq_len):
    data.append(df.Close[i: i + seq_len])


data = np.array(data)
train_size = int(0.8 * len(data))

X_train = torch.from_numpy(data[:train_size, :-1, :])
y_train = torch.from_numpy(data[:train_size, -1, :])
X_test = torch.from_numpy(data[train_size:, -1, :])
X_test = torch.from_numpy(data[train_size:, -1, :])

class PredictionModel(nn.Module):

    def __init__(self, input_dims, hidden_dims, num_layers, output_dims):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dims, output_dims)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dims)