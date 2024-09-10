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
import simpleaudio as sa
import random
import matplotlib.pyplot as plt
from math import *
from tqdm import tqdm
import schedulefree
import math

class WeatherDataset(Dataset):
    def __init__(self, ):
        self.frame = pd.read_csv("data/data.csv")

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
        #remove context_len from __len__ so that we choose start points 
        #where target = idx + context_len + output_len is in self.points_ds
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
    
def normalize(x, m = None, s = None):    
    if m is None or s is None:
        m = x.mean(dim=1)
        s = x.std(dim=1)
        
    return (x - m) / s, m, s

def un_normalize(x, m, s):        
    return (x * s) + m

def normalized_mse(series, pred, target):
    _, m, s = normalize(series)
        
    pred, _, _ = normalize(pred, m, s)
    target, _, _ = normalize(target, m, s)
        
    return nn.MSELoss()(pred, target)

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
    steps = 0
    cum_loss = 0

    for _, (series, target) in enumerate(progress_bar):
        optimizer.zero_grad()
        steps += 1
        
        series = series.to(device)
        target = target.to(device)
                   
        pred = model(series).squeeze()
        
        loss = normalized_mse(series, pred, target)
        losses.append(loss.item())

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
    output_len = 128
    context_len = 1024
    
    learning_rate = 3e-4
    batch_size = 16
    max_epochs = 1

    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    traffic_ds = WeatherDataset()

    ds_train = TimeSeriesDataset(traffic_ds, context_len, output_len, split = 'train')
    ds_test = TimeSeriesDataset(traffic_ds, context_len, output_len, split = 'test')

    dl_train = DataLoader(ds_train, batch_size = batch_size, shuffle = True)
    dl_test = DataLoader(ds_test, batch_size = batch_size, shuffle = True)
    
    model = MLPForecast(context_len, output_len)
    
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    losses = []
    
    for epoch in range(max_epochs):
        print("\n--------Epoch {}--------".format(epoch + 1))
        train(model, device, optimizer, dl_train)
        test(model, device, optimizer, dl_test)

    print("Training completed")
    








