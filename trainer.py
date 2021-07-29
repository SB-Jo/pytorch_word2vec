import torch
import numpy as np
from copy import deepcopy

class Trainer():
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        super().__init__()

    def _batchify(self, x, y, batch_size, random_split=True):
        if random_split:
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    def _train(self, x, y, config):
        self.model.train()
        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.criterion(y_hat_i, y_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print(f"Train Iteration({i+1}/{len(x)}): loss={float(loss_i):.3f}")

            total_loss += float(loss_i)

        return total_loss / len(x)

    

    def train(self, x, y, config):
        print('Word2Vec 학습 시작')
        for epoch in range(config.n_epochs):
            train_loss = self._train(x, y, config)

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch + 1,
                config.n_epochs,
                train_loss
            ))
        print('Word2Vec 학습 끝')
        return self.model