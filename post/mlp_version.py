#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

import os
import logging
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from math import *
from tqdm import tqdm
import schedulefree
import math

class WeatherDataset(Dataset):
    def __init__(self):
        self.frame = pd.read_csv('data.csv')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        data = self.frame.iloc[idx, 3]
        return torch.tensor(data, dtype=torch.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, points_ds, context_len, output_len, split='train'):
        full_length = len(points_ds)
        test_start = floor(full_length * 0.8)
        train_stop = test_start - context_len * 2
        
        if split == 'train':
            train_len = train_stop
            self.points = [points_ds[i] for i in range(train_len)]
        else:
            self.points = [points_ds[i] for i in range(test_start, full_length)]
        
        self.points = torch.stack(self.points)
        
        self.context_len = context_len
        self.output_len = output_len
        
    def __len__(self):
        return len(self.points) - self.context_len - self.output_len
    
    def __getitem__(self, idx):
        series = self.points[idx : idx + self.context_len]
        target = self.points[idx + self.context_len :
                             idx + self.context_len + self.output_len]
                        
        return series, target

class MLPForecast(nn.Module):
    def __init__(self, context_len, output_len):
        super().__init__()
                    
        hidden_dim = output_len * 4
        
        self.fc1 = nn.Linear(context_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_len)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x, m, s = normalize(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
                
        x = un_normalize(x, m, s)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout = 0.1, apply_ln = True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.residual = nn.Linear(input_dim, output_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(dropout)
        
        self.apply_ln = apply_ln
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        residual = self.residual(x)
        
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = x + residual
        if self.apply_ln:    
            x = self.layer_norm(x)
        
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 context_len,
                 patch_len,
                 big_patch_len,
                 output_patch_len,
                 d_model,
                 num_heads,
                 num_layers,
                 dropout):
        super().__init__()
        if context_len % patch_len != 0:
            raise Exception("context_len needs to be a multiple of patch_len")
            
        if context_len % big_patch_len != 0:
            raise Exception("context_len needs to be a multiple of patch_len*big_patch_multiple")
            
        self.d_model = d_model
        self.output_patch_len = output_patch_len
        
        self.patch_len = patch_len
        self.patches = context_len // patch_len
            
        self.big_patch_len = big_patch_len
        self.big_patches = context_len // big_patch_len
        
        self.patch_encoder = ResidualBlock(input_dim = patch_len * channels,
                                output_dim = d_model,
                                hidden_dim = d_model,
                                dropout = dropout,
                                apply_ln = True)
        self.big_patch_encoder = ResidualBlock(input_dim = big_patch_len * channels,
                                               output_dim = d_model,
                                               hidden_dim = d_model,
                                               dropout = dropout,
                                               apply_ln = False)
        self.patch_decoder = ResidualBlock(input_dim = d_model,
                                output_dim = output_patch_len * channels,
                                hidden_dim = d_model,
                                dropout = 0,
                                apply_ln = False)
        
        self.pos_embedding = self.sinusoidal_positional_embedding(self.patches + self.big_patches + 1, d_model).to(device)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def encode_patches(self, x, patches, patch_len, encoder):
        x = x.view(x.shape[0], patches, patch_len)
        encoded_patches = []
        
        for i in range(patches):
            patch_data = x[:, i, :].flatten(start_dim=1)
            encoded_patch = encoder(patch_data)
            encoded_patches.append(encoded_patch)
            
        return encoded_patches

    def forward(self, x):        
        x, m, s = normalize(x)
                
        encoded_patches = self.encode_patches(x, self.patches, self.patch_len, self.patch_encoder)
        encoded_big_patches = self.encode_patches(x, self.big_patches, self.big_patch_len, self.big_patch_encoder)

        x = torch.stack(encoded_patches + encoded_big_patches, dim=1)
                        
        start = self.start_token.expand(x.size(0), -1, self.d_model)
        x = torch.cat([start, x], dim=1)
        
        x = x + self.pos_embedding
                
        for layer in self.transformer_layers:
            tgt_mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)
            x = layer(x, x, tgt_mask=tgt_mask)
        
        x = x[:, -1, :]
        x = self.patch_decoder(x)
        x = x.view(x.shape[0], self.output_patch_len, self.channels)
        
        x = un_normalize(x, m, s)
        
        return x
    
    def sinusoidal_positional_embedding(self, num_positions, d_model):
        position = torch.arange(0, num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pos_embedding = torch.zeros(num_positions, d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        pos_embedding = pos_embedding.unsqueeze(0)
        
        return pos_embedding
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
def normalize(x, m = None, s = None):    
    if m is None or s is None:
        m = x.mean(dim=1, keepdim=True)
        s = x.std(dim=1, keepdim = True)
    
    return (x - m) / s, m, s

def un_normalize(x, m, s):        
    return (x * s) + m

def normalized_mse(series, pred, target):
    _, m, s = normalize(series)
        
    pred, _, _ = normalize(pred, m, s)
    target, _, _ = normalize(target, m, s)
        
    return nn.MSELoss()(pred, target)


def plot_dataset(ds, prediction=None):
    plt.figure(figsize=(16, 5), dpi=200)
    
    x = np.arange(context_len + output_len)
    last_val = np.array([ds[0][-1].numpy()])
    future = np.concatenate((last_val, ds[1].numpy()))
    
    plt.plot(x[:context_len], ds[0].numpy(), color='blue', label='Input')
    plt.plot(x[context_len - 1:], future, color='orange', label='Target')
    
    if prediction is not None:
        prediction = np.concatenate((last_val, prediction.numpy()))
        plt.plot(x[context_len - 1:], prediction, color='green', label='Prediction')
    
    plt.xlabel("Reading", fontsize=12)
    plt.ylabel("Temperature (C)", fontsize=12)
    if prediction is not None:
        plt.title('MLP Prediction Example', fontsize=14)
    else:
        plt.title('Weather dataset example', fontsize=14)
    
    plt.legend(fontsize=12)
    
    plt.tight_layout() 
    
    if prediction is not None:
        plt.savefig("mlp_prediction.png", format="png", dpi=200)
    else:
        plt.savefig("dataset_example.png", format="png", dpi=200)

    plt.show()

    



"""
def forecast(data, model, steps=10):
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            y = data[-context_len:].unsqueeze(0)
            
            next_y = model(y)
            
            data = torch.cat((data, next_y))
            
        data = data.detach().cpu().numpy()
            
        x = np.arange(len(data))    
        
        plt.figure(figsize=(16, 6))
        
        plt.plot(x[:context_len], data[:context_len], color='blue', label='Recorded')
        plt.plot(x[context_len:], data[context_len:], color='orange', label='Forecast')
        
        plt.title('Forecast') 
        plt.legend()
        
        plt.show()
"""

def train(model, device, optimizer, dataloader):
    model.train()
    optimizer.train()
    
    progress_bar = tqdm(dataloader, desc="Training", leave=True)
    cum_loss = 0

    for step, (series, target) in enumerate(progress_bar):
        optimizer.zero_grad()
        
        series = series.to(device)
        target = target.to(device)
                   
        pred = model(series).squeeze()
        
        loss = normalized_mse(series, pred, target)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
            
        cum_loss += loss.item()
        progress_bar.set_postfix(running_loss = cum_loss/(step + 1))
    
def test(model, device, optimizer, dataloader):
    model.eval()
    optimizer.eval()
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=True)
    
    cum_loss = 0
    
    with torch.no_grad():
        for step, (series, target) in enumerate(progress_bar):           
            series = series.to(device)
            target = target.to(device)
            
            pred = model(series).squeeze()
            loss = normalized_mse(series, pred, target)
            
            cum_loss += loss.item()
            progress_bar.set_postfix(running_loss = cum_loss/(step + 1))
                    
    print("Validation MSE: {}".format(cum_loss/len(dataloader)))

if __name__ == '__main__':
    output_len = 128
    context_len = 1024
    
    learning_rate = 3e-4
    batch_size = 32
    max_epochs = 1

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    weather_ds = WeatherDataset()

    ds_train = TimeSeriesDataset(weather_ds, context_len, output_len, split = 'train')
    ds_test = TimeSeriesDataset(weather_ds, context_len, output_len, split = 'test')
    
    plot_dataset(ds_test[0])

    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)
    dl_test = DataLoader(ds_test, batch_size = batch_size, shuffle = True)
    
    model = MLPForecast(context_len, output_len)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    losses = []
    
    for epoch in range(max_epochs):
        print("--------Epoch {}--------".format(epoch + 1))
        train(model, device, optimizer, dl_train)
        test(model, device, optimizer, dl_test)

    print("Training completed")
    
    print("Model parameters: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    model_output = model(ds_test[0][0].unsqueeze(0).to(device)).squeeze().detach().cpu()
    
    plot_dataset(ds_test[6000], model_output)
    








