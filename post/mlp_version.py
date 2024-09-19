#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import schedulefree
import math


class WeatherDataset(Dataset):
    def __init__(self):
        self.frame = pd.read_csv("data.csv")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        data = self.frame.iloc[idx, 3]
        return torch.tensor(data, dtype=torch.float32)


class TimeSeriesDataset(Dataset):
    def __init__(self, points_ds, context_len, output_len, split="train"):
        full_length = len(points_ds)
        test_start = math.floor(full_length * 0.8)
        train_stop = test_start - context_len * 2

        if split == "train":
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
        target = self.points[
            idx + self.context_len : idx + self.context_len + self.output_len
        ]

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
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1, apply_ln=True):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.residual = nn.Linear(input_dim, output_dim, bias=False)
        self.gelu = nn.GELU(approximate="tanh")
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


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, num_positions, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.num_positions = num_positions
        self.d_model = d_model
        self.register_buffer("pos_embedding", self.create_positional_embedding())

    def create_positional_embedding(self):
        position = torch.arange(0, self.num_positions).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model)
        )

        pos_embedding = torch.zeros(self.num_positions, self.d_model)
        pos_embedding[:, 0::2] = torch.sin(position * div)
        pos_embedding[:, 1::2] = torch.cos(position * div)

        return pos_embedding.unsqueeze(0)

    def forward(self):
        return self.pos_embedding


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        context_len,
        patch_len,
        output_patch_len,
        output_len,
        d_model,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.output_patch_len = output_patch_len

        self.patch_len = patch_len
        self.patches = context_len // patch_len

        self.output_patches = output_len // output_patch_len
        self.tokens = self.patches + self.output_patches

        self.patch_encoder = ResidualBlock(
            input_dim=patch_len,
            output_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout,
            apply_ln=True,
        )

        self.patch_decoder = ResidualBlock(
            input_dim=d_model,
            output_dim=output_patch_len,
            hidden_dim=d_model,
            dropout=0,
            apply_ln=False,
        )

        self.output_tokens = nn.Parameter(torch.randn(1, self.output_patches, d_model))
        self.pos_embedding = SinusoidalPositionalEmbedding(self.tokens, d_model)

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.register_buffer("tgt_mask", self.causal_mask(self.tokens))

    def encode_patches(self, x, patches, patch_len, encoder):
        x = x.view(x.shape[0], patches, patch_len)
        encoded_patches = []

        for i in range(patches):
            patch_data = x[:, i, :].flatten(start_dim=1)
            encoded_patch = encoder(patch_data)
            encoded_patches.append(encoded_patch)

        return encoded_patches

    def decode_patches(self, x, output_patches, output_patch_len, decoder):
        decoded_patches = []

        for i in range(1, output_patches + 1):
            decoded_patches.append(decoder(x[:, -i, :]))

        return torch.cat(decoded_patches, dim=1)

    def forward(self, x):
        x, m, s = normalize(x)

        encoded_patches = self.encode_patches(
            x, self.patches, self.patch_len, self.patch_encoder
        )

        x = torch.stack(encoded_patches, dim=1)
        x = torch.cat((x, self.output_tokens.repeat(x.shape[0], 1, 1)), dim=1)

        x = x + self.pos_embedding()

        for layer in self.transformer_layers:
            x = layer(x, src_mask=self.tgt_mask, is_causal=True)

        x = self.decode_patches(
            x, self.output_patches, self.output_patch_len, self.patch_decoder
        )

        # x = x[:, -1, :]
        # x = self.patch_decoder(x)

        x = un_normalize(x, m, s)

        return x

    def causal_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask


def normalize(x, m=None, s=None):
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


def plot_dataset(ds, idx, title, prediction=None):
    plt.figure(figsize=(16, 3), dpi=140)

    x = np.arange(context_len + len(ds[idx][1]))
    last_val = np.array([ds[idx][0][-1].numpy()])

    future = np.concatenate((last_val, ds[idx][1].numpy()))

    plt.plot(x[:context_len], ds[idx][0].numpy(), color="blue", label="Input")
    plt.plot(x[context_len - 1 :], future, color="orange", label="Target")

    if prediction is not None:
        prediction = np.concatenate((last_val, prediction.numpy()))

        plt.plot(x[context_len - 1 :], prediction, color="green", label="Prediction")

    plt.ylabel("Temperature (C)", fontsize=12)
    plt.title(title, fontsize=14)

    plt.legend(fontsize=12)

    plt.tight_layout()

    plt.savefig(title.lower().replace(" ", "_") + ".png", format="png", dpi=140)

    plt.show()


def autoregressive_forecast(model, title="", steps=10, start_idx=3000):
    ds = TimeSeriesDataset(weather_ds, context_len, output_len * steps, split="test")
    data = ds[start_idx][0].to(device)

    model.eval()
    with torch.no_grad():
        for step in range(steps):
            y = data[-context_len:].unsqueeze(0)

            next_y = model(y).squeeze()

            data = torch.cat((data, next_y))

        data = data.detach().cpu().squeeze()[context_len:]

        plot_dataset(
            ds, 1000, title=title, prediction=data
        )

def compare_multi_token():
    full_res = []
    repeats = 6

    for multiple in [2, 3, 4, 5]:
        output_len = output_patch_len * multiple
        
        ds_train = TimeSeriesDataset(weather_ds, context_len, output_len, split="train")
        ds_test = TimeSeriesDataset(weather_ds, context_len, output_len, split="test")

        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)
        
        mul_res = []
        for mode in ["multi_token", "long_output"]:
            
            losses = []
            
            for repeat in range(repeats):
                model = TimeSeriesTransformer(
                    context_len,
                    patch_len,
                    output_len if mode == "long_output" else output_patch_len,
                    output_len,
                    d_model,
                    num_heads,
                    num_layers,
                    dropout,
                )
                model = model.to(device)
                optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
                
                final_loss = 0
                
                print("")
                print(f"Starting training with repeat: {repeat+1}, multiple: {multiple}, mode: {mode}")
                print("")
                                
                for epoch in range(max_epochs):
                    print("--------Epoch {}--------".format(epoch + 1))
                    train(model, device, optimizer, dl_train)
                    final_loss = test(model, device, optimizer, dl_test)
                    
                losses.append(final_loss)
                
            mul_res.append(([losses], np.array(losses).mean(), np.array(losses).std()/math.sqrt(repeats), mode, multiple))
            
        full_res.append(mul_res)
        
    # Extract unique multiples
    multiples = sorted(list(set(entry[0][4]*128 for entry in full_res)))  # [2, 3, 4, 5]
    multiples = ["256 = 128×2", "384 = 128×3", "512 = 128×4", "640 = 128×5"]
    
    # Initialize dictionaries to hold means and stds for each mode
    means = {'multi_token': [], 'long_output': []}
    stds = {'multi_token': [], 'long_output': []}
    
    # Populate the means and stds dictionaries
    for mul_res in full_res:
        for res in mul_res:
            _, mean, std, mode, multiple = res
            means[mode].append(mean)
            stds[mode].append(std)
    
    # Define bar properties
    bar_width = 0.35
    index = np.arange(len(multiples))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    
    # Plot bars for 'multi_token'
    bars1 = ax.bar(index - bar_width/2, means['multi_token'], bar_width, yerr=stds['multi_token'],
                   label='Multi-token', capsize=5, color='skyblue', edgecolor='black')
    
    # Plot bars for 'long_output'
    bars2 = ax.bar(index + bar_width/2, means['long_output'], bar_width, yerr=stds['long_output'],
                   label='Long output', capsize=5, color='salmon', edgecolor='black')
    
    # Add labels and title
    ax.set_xlabel('Output length', fontsize=12)
    ax.set_ylabel('Mean MSE', fontsize=12)
    ax.set_title('Multi-token vs. long output mean MSE', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(multiples)
    ax.legend()
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    
    def autolabel(bars, stds_list):
        """Attach a text label above each bar displaying its height plus std."""
        for bar, std in zip(bars, stds_list):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height + std),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(bars1, stds['multi_token'])
    autolabel(bars2, stds['long_output'])
            
    plt.tight_layout()

    plt.savefig("multi-token_vs_output_len.png", format="png", dpi=140)
    
    plt.show()
        
        

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

        loss.backward()
        optimizer.step()

        cum_loss += loss.item()
        progress_bar.set_postfix(running_loss=cum_loss / (step + 1))


def test(model, device, optimizer, dataloader):
    model.eval()
    optimizer.eval()

    progress_bar = tqdm(dataloader, desc="Testing", leave=True)

    cum_loss = 0

    with torch.no_grad():
        for step, (series, target) in enumerate(progress_bar):
            series = series.to(device)
            target = target.to(device)

            pred = model(series).squeeze()
            loss = normalized_mse(series, pred, target)

            cum_loss += loss.item()
            progress_bar.set_postfix(running_loss=cum_loss / (step + 1))

    val_mse = cum_loss / len(dataloader)
    print(f"Training MSE: {val_mse}")
    return val_mse


if __name__ == "__main__":
    patch_len = 32
    output_patch_len = 128

    context_len = 2048
    d_model = 256
    num_heads = 4
    num_layers = 4
    dropout = 0.1

    model_type = "Transformer"
    #model_type = 'MLP'
    output_patches = 4
    ar_steps = 5
    
    output_len = output_patch_len * output_patches
    
    learning_rate = 3e-4
    batch_size = 32 * 2
    max_epochs = 5

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    
    weather_ds = WeatherDataset()
    
    ds_train = TimeSeriesDataset(weather_ds, context_len, output_len, split="train")
    ds_test = TimeSeriesDataset(weather_ds, context_len, output_len, split="test")

    plot_dataset(ds_test, 3000, "Weather dataset example")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    if model_type == "MLP":
        model = MLPForecast(context_len, output_len)
    else:
        model = TimeSeriesTransformer(
            context_len,
            patch_len,
            output_patch_len,
            output_len,
            d_model,
            num_heads,
            num_layers,
            dropout,
        )

    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for epoch in range(max_epochs):
        print("--------Epoch {}--------".format(epoch + 1))
        train(model, device, optimizer, dl_train)
        test(model, device, optimizer, dl_test)

    print("Training completed")

    print(
        "Model parameters: {:,}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    for i, example_id in enumerate([6000, 0, 3000]):
        multitoken = " multi-token" if output_patches > 1 else ""
        
        model_output = (
            model(ds_test[example_id][0].unsqueeze(0).to(device))
            .squeeze()
            .detach()
            .cpu()
        )
        plot_dataset(
            ds_test,
            example_id,
            f"{model_type}{multitoken} prediction example {i + 1}",
            model_output,
        )
        
    autoregressive_forecast(model, title=f"{ar_steps} step {model_type}{multitoken} forecast", steps=ar_steps, start_idx=1000)
    
    compare_multi_token()

            
            
        
