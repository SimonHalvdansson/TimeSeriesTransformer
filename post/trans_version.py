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

class GenericCSVDataset(Dataset):
    def __init__(self, path, valid_channels):
        self.frame = pd.read_csv(path)
        self.valid_channels = valid_channels

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        data = self.frame.iloc[idx]
        data = data.iloc[self.valid_channels].astype(float).to_numpy()
            
        return torch.tensor(data, dtype=torch.float32)

class TimeSeriesDataset(Dataset):
    def __init__(self, points_ds, context_len, output_len, split='train'):
        full_length = len(points_ds)
        validation_start = floor(full_length * 0.8)
        train_stop = validation_start - context_len * 2
        
        if split == 'train':
            train_len = train_stop
            self.points = [points_ds[i] for i in range(train_len)]
        else:
            test_start = validation_start
            self.points = [points_ds[i] for i in range(test_start, full_length)]
        
        self.points = torch.stack(self.points)
        
        self.context_len = context_len
        self.output_len = output_len
        self.total_channels = len(points_ds.valid_channels)
        
    def __len__(self):
        #remove context_len from __len__ so that we choose start points where
        #where target = idx + context_len + output_len is in self.points_ds
        return len(self.points) - self.context_len - self.output_len
    
    def __getitem__(self, idx):
        channel = random.randint(0, self.total_channels - 1)
                
        series = self.points[idx : idx + self.context_len, channel]
        target = self.points[idx + self.context_len : idx + self.context_len + self.output_len, channel]
                        
        return series, target


class ResidualBlock(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 dropout = 0.1,
                 apply_ln = True):
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
            patch_data = x[:, i, :, :].flatten(start_dim=1)
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
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    if m is None or s is None:
        m = x.mean(dim=1, keepdim=True)
        s = x.std(dim=1, keepdim=True)
        
    return (x - m) / s, m, s

def un_normalize(x, m, s):        
    return (x * s) + m

def normalized_mse(series, pred, target):
    _, m, s = normalize(series)
        
    pred, _, _ = normalize(pred, m, s)
    target, _, _ = normalize(target, m, s)
        
    return nn.MSELoss()(pred, target)

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


def train(model, device, optimizer, dataloader):
    model.train()
    optimizer.train()
    
    progress_bar = tqdm(dataloader, desc="Training", leave=True)
    steps = 0
    cum_loss = 0
    
    scaler = torch.amp.GradScaler() if use_autocast else None

    for _, (series, target) in enumerate(progress_bar):
        optimizer.zero_grad()
        steps += 1
        
        series = series.to(device)
        target = target.to(device)
                   
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_autocast):
            pred = model(series).squeeze()
            
            loss = normalized_mse(series, pred, target)
            losses.append(loss.item())

        if use_autocast:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        cum_loss += loss.item()
        progress_bar.set_postfix(running_loss = cum_loss/steps)

    running_avg = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')

    plt.plot(np.log(np.array(losses)), label='Loss')

    if len(losses) > window_size:
        plt.plot(range(window_size - 1, len(losses)), np.log(np.array(running_avg)), label='Running avg.', color='orange')
    
    plt.title("Log loss")
    plt.legend()
    plt.show()
    
def test(model, device, optimizer, dataloader):
    model.eval()
    optimizer.eval()
    
    progress_bar = tqdm(dataloader, desc="Validating", leave=True)
    
    steps = 0
    cum_loss = 0
    
    with torch.no_grad():
        for _, (series, target) in enumerate(progress_bar):
            steps += 1
            
            series = series.to(device)
            target = target.to(device)
            
            pred = model(series).squeeze()
            loss = normalized_mse(series, pred, target)
            
            cum_loss += loss.item()
            progress_bar.set_postfix(running_loss = cum_loss/steps)
                    
    print("\nValidation MSE: {}".format(cum_loss/steps))

if __name__ == '__main__':
    patch_len = 32
    output_patch_len = 128
    big_patch_len = patch_len * 8
    context_len = patch_len * 32
    d_model = 256
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    
    learning_rate = 3e-4
    batch_size = 16
    max_epochs = 1
    window_size = 200
    use_mps = True
    use_autocast = True
    run = True

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_float32_matmul_precision('high')
    elif use_mps and torch.backends.mps.is_available():
        device = 'mps'
    
    use_autocast = use_autocast and device == 'cuda'

    #weather_ds = GenericCSVDataset(path = "data/data.csv", valid_channels = [3, 4])
    traffic_ds = GenericCSVDataset(path = "data/traffic/traffic.csv", valid_channels = list(range(1, 862)))

    ds_train = TimeSeriesDataset(traffic_ds, context_len, output_patch_len, split = 'train')
    ds_test = TimeSeriesDataset(traffic_ds, context_len, output_patch_len, split = 'test')

    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)
    dl_test = DataLoader(ds_test, batch_size = batch_size, shuffle = True)
    
    model = TimeSeriesTransformer(context_len,
                                  patch_len,
                                  big_patch_len,
                                  output_patch_len,
                                  d_model,
                                  num_heads,
                                  num_layers,
                                  dropout)
    #model = MLPForecast(context_len, patch_len) #validation MSE: ?
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    losses = []
    
    if not run:
        max_epochs = 0
    for epoch in range(max_epochs):
        print("\n--------Epoch {}--------".format(epoch + 1))
        
        train(model, device, optimizer, dl_train)
        test(model, device, optimizer, dl_test)
        

    print("Training completed")
    
    #forecast(ds_test[123][0], model, 10)








