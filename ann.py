import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from soil_dataset import SoilDataset
from time import time


class ANN(nn.Module):
    def __init__(self, device, train_x, train_y, test_x, test_y, validation_x, validation_y):
        super().__init__()
        torch.manual_seed(1)
        self.verbose = False
        self.TEST = False
        self.device = device
        self.train_ds = SoilDataset(train_x, train_y)
        self.test_ds = SoilDataset(test_x, test_y)
        self.validation_ds = SoilDataset(validation_x, validation_y)
        self.num_epochs = 2000
        self.batch_size = 3000
        self.lr = 0.01
        x_size = validation_x.shape[1]

        self.linear = nn.Sequential(
            nn.Linear(x_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.linear(x)
        return x

    def train_model(self):
        if self.TEST:
            return
        self.train()
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss(reduction='sum')
        dataloader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        total_batch = len(dataloader)
        for epoch in range(self.num_epochs):
            for batch_number, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self(x)
                y_hat = y_hat.reshape(-1)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if self.verbose:
                    r2_test = r2_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
                    y_all, y_hat_all = self.evaluate(self.validation_ds)
                    r2_validation = r2_score(y_all, y_hat_all)
                    print(f'Epoch:{epoch + 1} (of {self.num_epochs}), Batch: {batch_number+1} of {total_batch}, '
                          f'Loss:{loss.item():.3f}, R2_TEST: {r2_test:.3f}, R2_Validation: {r2_validation:.3f}')

    def evaluate(self, ds):
        batch_size = 30000
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        y_all = np.zeros(0)
        y_hat_all = np.zeros(0)

        for (x, y) in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            y_hat = y_hat.reshape(-1)
            y = y.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()

            y_all = np.concatenate((y_all, y))
            y_hat_all = np.concatenate((y_hat_all, y_hat))

        return y_all, y_hat_all

    def test(self):
        self.eval()
        self.to(self.device)
        y_all, y_hat_all = self.evaluate(self.test_ds)
        return y_hat_all

