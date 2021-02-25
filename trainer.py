# %%
import torch
from torch import nn

import numpy as np
import time

import matplotlib.pyplot as plt


# %%

class Trainer:
    def __init__(self, model, X, y,
                 optimizer,
                 criterion,
                 scheduler=None,
                 train_test_split=.8,
                 train_val_split=.9):

        self.train_test_split = train_test_split
        self.train_val_split = train_val_split
        self.criterion = criterion

        # model
        self.model = model

        # optimizer & scheduler
        self.optim = optimizer
        self.scheduler = scheduler

        # initialise variables to store loss
        self.test_loss = None
        self.train_loss_all = []
        self.val_loss_all = []

        # set up data set and dataloaders
        self.data = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        self.train_loader, self.validation_loader, self.test_loader = None, None, None

        # compute sizes of splits
        # train, test split
        self.len_train = int(len(self.data) * self.train_test_split)
        self.len_test = len(self.data) - self.len_train
        assert (self.len_test + self.len_train) == len(self.data), 'Train/test split incorrect'

        # we perform this split ONCE at the beginning to test the model after training with the test set
        self.train_set, self.test_set = torch.utils.data.random_split(self.data, [self.len_train, self.len_test])
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=1, shuffle=False)

        # train, validation split
        # we generate a new train, validation split in each epoch
        # update len_train according to the train/validation split
        self.len_val = int(self.len_train * (1 - self.train_val_split))
        self.len_train = self.len_train - self.len_val

        print('Number of data points:')
        print(f'\tTraining: {self.len_train} ({self.len_train / len(self.data) * 100:.2f}%)')
        print(f'\tTest: {self.len_test} ({self.len_test / len(self.data) * 100:.2f}%)')
        print(f'\tValidation: {self.len_val} ({self.len_val / len(self.data) * 100:.2f}%)')

    def train_model(self, epochs=10, batch_size=128, shuffle=True):
        print(f'Start training: {time.asctime()}')
        print(f'\tepochs: {epochs}, batch_size: {batch_size}, criterion: {self.criterion._get_name()}')

        for epoch in range(epochs):
            tic = time.time()
            # generate dataloaders
            self.generator_dataloaders(batch_size, shuffle=shuffle)

            # train
            self.model.train()
            self.train_loss = []
            for x, y in self.train_loader:
                # predict
                y_hat = self.model(x)

                # compute loss
                loss = self.criterion(y_hat, y)

                # optimise
                self.optim.zero_grad()  # zero gradients, otherwise we accumulate gradients
                loss.backward()
                self.optim.step()

                # track loss
                self.train_loss.append(loss.detach().item())

            self.train_loss_all.append(self.train_loss.copy())

            # validate
            self.model.eval()
            self.val_loss = []
            for x, y in self.validation_loader:
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                self.val_loss.append(loss.detach().item())
            self.val_loss_all.append(self.val_loss.copy())

            # adjust optimiser params
            if self.scheduler:
                self.scheduler.step()

            toc = time.time()
            print(
                f'[{epoch + 1}/{epochs}, training loss: {np.mean(self.train_loss):.5f} ({np.std(self.train_loss):.5f}) [N={len(self.train_loss)}], validation loss: {np.mean(self.val_loss):.5f} ({np.std(self.val_loss):.5f}) [N={len(self.val_loss)}], duration {(toc - tic) / 60:.3f} min]')

        # test
        self.model.eval()
        self.test_loss = []
        for x, y in self.test_loader:
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            self.test_loss.append(loss.detach().item())
        print('------------------------------------------------------------------------------')
        # print(f'Test loss: {self.test_loss:.5f} ({np.std(self.test_loss):.5f}) [N={len(self.test_loss)}], average last training loss: {np.mean(self.train_loss):.5f} ({np.std(self.train_loss)}) [N={len(self.train_loss)}]')
        print(f'Test loss: {np.mean(self.test_loss):.5f} ({np.std(self.test_loss):.5f}) [N={len(self.test_loss)}]')
        print('------------------------------------------------------------------------------')
        print('Training finished.')

    def generator_dataloaders(self, batch_size=128, shuffle=True):
        """
            Generates loaders for training and validation.
            Can be used to generate new loaders in each epoch.
        """
        train, validate = torch.utils.data.random_split(self.train_set, [self.len_train, self.len_val], )
        self.train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=shuffle)
        self.validation_loader = torch.utils.data.DataLoader(dataset=validate, batch_size=batch_size, shuffle=False)

    def plot_loss(self, save=None):
        fig, ax = plt.subplots()
        train_means = [np.mean(i) for i in self.train_loss_all]
        train_stds = [np.std(i) for i in self.train_loss_all]

        val_means = [np.mean(i) for i in self.val_loss_all]
        val_stds = [np.std(i) for i in self.val_loss_all]

        test_means = np.mean(self.test_loss)
        test_stds = np.std(self.test_loss)

        ax.plot(range(len(train_means)), train_means, marker='o', linestyle='--', label='train')
        ax.plot(range(len(val_means)), val_means, marker='o', linestyle='--', label='validate')
        ax.plot([0, len(train_means)-1], [test_means,test_means], marker=None, linestyle='--', label='test')
        ax.set_title('Loss')
        ax.set_xlabel('epochs')

        nticks = min(20,len(train_means))
        ticks = np.linspace(0, len(train_means), nticks).astype(int)
        ax.set_xticks(ticks)

        ax.set_xticklabels(ticks + 1)
        ax.legend()
        fig.show()
        
        if save:
            fig.savefig(save)


# %%
