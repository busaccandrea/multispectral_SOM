import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxUnpool1d
from torch.optim import Adam, optimizer
from utilities import check_existing_folder

class Conv1DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv1DAutoencoder, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            # first conv
            nn.Conv1d(in_channels=1, out_channels=64, padding=2, kernel_size=5), #input size: 2048
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 

            nn.Conv1d(in_channels=64, out_channels=64, padding=2, kernel_size=5), #input size from maxpool: 1024
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=64, padding=2, kernel_size=5),#input size from maxpool: 512
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),


            nn.Conv1d(in_channels=64, out_channels=64, padding=2, kernel_size=5),#input size from maxpool: 256
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=1, padding=2, kernel_size=5),#input from maxpool: 128
            nn.ReLU() # output size: 128
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, padding=2, kernel_size=5),#input size = 128
            nn.MaxUnpool1d(kernel_size=2, stride=2),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=64, padding=2, kernel_size=5), # input from maxunpool 256
            nn.MaxUnpool1d(kernel_size=2, stride=2),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=64, padding=2, kernel_size=5), # input from maxunpool 512
            nn.MaxUnpool1d(kernel_size=2, stride=2),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=64, padding=2, kernel_size=5), # input from maxunpool 1024
            nn.MaxUnpool1d(kernel_size=2, stride=2)

            #output size = 2048
        )
        self.losses = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def forward(self, x):
        return self.decoder(self.encoder(x))

    def train_autoencoder(self, train_dataloader, lr=0.01, epochs=10):
        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for e in range(epochs):
            print('epoch:', e)
            self.train()
            
            with torch.set_grad_enabled(True):
                for _, batch in enumerate(train_dataloader):
                    x = batch[0].to(self.device)

                    features_of_x = self.encoder(x)

                    decoded = self.decoder(features_of_x)
                    
                    # MSE between original and decoded versions of x
                    mse = criterion(decoded, x)

                    # backpropagation
                    mse.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    self.losses.append(mse.item())
            check_existing_folder('data/Conv1DAutoencoder/checkpoints/')
            torch.save(self.state_dict(), 'data/Conv1DAutoencoder/checkpoints/%s.pth'%('Conv1DAutoencoder'))
            