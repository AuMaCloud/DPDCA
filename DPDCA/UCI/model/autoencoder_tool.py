# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


###########################################################################
# The following classes from the codebase are adapted for different datasets:
# https://github.com/astorfi/differentially-private-cgan/blob/master/UCI/autoencoder.py
###########################################################################
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        n_channels_base = 4
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=4, padding=0,),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=4, padding=0),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=1, kernel_size=7, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x.view(-1, 1, x.shape[1]))
        x = self.decoder(x)

        return torch.squeeze(x, dim=1)

    def decode(self, x):
        x = self.decoder(x)

        return torch.squeeze(x, dim=1)


def autoencoder_loss(x_output, y_target):
    MSEloss = nn.MSELoss(reduction='sum')

    return MSEloss(x_output, y_target)


def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def set_seed(seed = 2000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


