#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchaudio
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

class SpectrogramEncoder(nn.Module):
    def __init__(self, output_dim, n_fft = 256):
        super(SpectrogramEncoder, self).__init__()
        
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=n_fft//2,
            power=2.0
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,   
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False
        )
        
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=output_dim,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False
        )
        self.relu = nn.ReLU()
        
        self.skip_proj = nn.Conv2d(
            in_channels=64,
            out_channels=output_dim,
            kernel_size=(1, 1),
            bias=False
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.spectrogram(x)
        x = self.amp_to_db(x)

        x = self.conv1(x)
        x = self.relu(x)

        skip = self.skip_proj(x)        
        x = self.conv2(x)
        
        x += skip
        
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.permute(0, 3, 2, 1)
        x = self.layer_norm(x)
        x = x.mean(dim=2)
        
        x = self.mlp(x)
        
        return x
        

class TimeSeriesTransformer(nn.Module):
    def __init__(self,
                 context_len,
                 patch_len,
                 output_patch_len,
                 d_model,
                 n_fft,
                 num_heads,
                 num_layers,
                 dropout):
        super().__init__()
        if context_len % patch_len != 0:
            raise Exception("context_len needs to be a multiple of patch_len")

        self.d_model = d_model
        self.output_patch_len = output_patch_len
        
        self.patch_len = patch_len
        self.patches = context_len // patch_len
        
        #self.spec_encoder = SpectrogramEncoder(output_dim = d_model, n_fft = n_fft)
        self.spec_tokens = 2 * context_len // n_fft + 1
                
        self.patch_encoder = ResidualBlock(input_dim = patch_len,
                                output_dim = d_model,
                                hidden_dim = d_model,
                                dropout = dropout,
                                apply_ln = True)

        self.patch_decoder = ResidualBlock(input_dim = d_model,
                                output_dim = output_patch_len,
                                hidden_dim = d_model,
                                dropout = 0,
                                apply_ln = False)
        
        self.total_tokens = self.patches + self.spec_tokens*0
        self.pos_embedding = self.sinusoidal_positional_embedding(self.total_tokens, d_model).to(device)
        self.tgt_mask = self.generate_square_subsequent_mask(self.total_tokens).to(device)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        

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
        
        #spectrogram_tokens = self.spec_encoder(x)
                        
        encoded_patches = self.encode_patches(x, self.patches, self.patch_len, self.patch_encoder)

        x = torch.stack(encoded_patches, dim=1)
                           
        #x = torch.cat((x, spectrogram_tokens), dim = 1)
                
        x = x + self.pos_embedding
                
        for layer in self.transformer_layers:
            x = layer(x, x, tgt_mask=self.tgt_mask, tgt_is_causal=True)
        
        x = x[:, -1, :]
        x = self.patch_decoder(x)
        x = x.view(x.shape[0], self.output_patch_len)
        
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


def plot_dataset(ds, idx, title, prediction=None):
    plt.figure(figsize=(16, 3), dpi=200)
    
    x = np.arange(context_len + len(ds[idx][1]))
    last_val = np.array([ds[idx][0][-1].numpy()])
    
    future = np.concatenate((last_val, ds[idx][1].numpy()))
    
    plt.plot(x[:context_len], ds[idx][0].numpy(), color='blue', label='Input')
    plt.plot(x[context_len - 1:], future, color='orange', label='Target')
    
    if prediction is not None:
        prediction = np.concatenate((last_val, prediction.numpy()))
        
        plt.plot(x[context_len - 1:], prediction, color='green', label='Prediction')
    
    plt.ylabel("Temperature (C)", fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.legend(fontsize=12)
    
    plt.tight_layout() 
    
    plt.savefig(title.lower().replace(' ', '_') + '.png', format='png', dpi=200)
    
    plt.show()


def autoregressive_forecast(model, steps = 10, start_idx = 3000):
    #new dataset with longer output_len
    ds = TimeSeriesDataset(weather_ds, context_len, output_patch_len * steps, split = 'test')
    
    data = ds[start_idx][0].to(device)
    
    model.eval()
    with torch.no_grad():
        for step in range(steps):
            y = data[-context_len:].unsqueeze(0)
            
            next_y = model(y).squeeze()
            
            data = torch.cat((data, next_y))
            
        data = data.detach().cpu().squeeze()[context_len:]
            
        plot_dataset(ds, 1000, title=f'{steps} step {model_type} forecast', prediction = data)
    


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
    
    val_mse = cum_loss/len(dataloader)
    print(f'Validation MSE: {val_mse}')
    return val_mse

if __name__ == '__main__':
    patch_len = 32
    output_patch_len = 128
    output_len = output_patch_len * 1
    context_len = 1024*2
    d_model = 256
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    n_fft = 256
    
    model_type = 'Transformer'
    #model_type = 'MLP'
    
    learning_rate = 3e-4
    batch_size = 32
    max_epochs = 1

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    weather_ds = WeatherDataset()

    ds_train = TimeSeriesDataset(weather_ds, context_len, output_patch_len, split = 'train')
    ds_test = TimeSeriesDataset(weather_ds, context_len, output_patch_len, split = 'test')
    
    plot_dataset(ds_test, 3000, "Weather dataset example")

    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)
    dl_test = DataLoader(ds_test, batch_size = batch_size, shuffle = True)
    
    if model_type == 'MLP':
        model = MLPForecast(context_len, output_len)
    else:    
        model = TimeSeriesTransformer(context_len, patch_len, output_patch_len, d_model, n_fft, num_heads, num_layers, dropout)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    losses = []
    
    for epoch in range(max_epochs):
        print("--------Epoch {}--------".format(epoch + 1))
        train(model, device, optimizer, dl_train)
        test(model, device, optimizer, dl_test)

    print("Training completed")
    
    print("Model parameters: {:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    for i, example_id in enumerate([6000, 0, 3000]):
        model_output = model(ds_test[example_id][0].unsqueeze(0).to(device)).squeeze().detach().cpu()
        plot_dataset(ds_test, example_id, f'{model_type} Prediction Example {i + 1}', model_output)


    autoregressive_forecast(model, steps = 5, start_idx = 1000)








