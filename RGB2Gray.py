from random import sample
from matplotlib import animation
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.optim import Adam, optimizer
from sklearn.metrics import accuracy_score
from utilities import check_existing_folder
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

""" modello per la separazione del pentimento dal visibile.
                                        real Pb Rx |-->
    "train": RGB (no pentimenti) -> CNN ->   Pb Rx |--> MSE -.
              ^----------------------------------------------'

    application: RGB -> f(x) -> Rx' |->
                     pentimento Rx  |-> ( - ) --> rx without pentimento
 """

class RGB2Gray(nn.Module):
    def __init__(self):
        # seven conv layers
        super(RGB2Gray, self).__init__()
        self.hidden_layers = nn.Sequential(
                    # L0
                    # 3 canali in input (RGB), 128 in output
                    nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(),
                    # L1
                    # 128 canali in input (il livello precedente ha 128 canali di output), 128 in output
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(),
                    # L2
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(),
                    # L3
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=128),
                    nn.ReLU(),

                    # L4
                    # 128 canali in input (il livello precedente ha 128 canali di output), 256 in output
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),

                    # L5
                    # 256 canali in input (il livello precedente ha 256 canali di output), 256 in output
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),
                    # L6
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=256),
                    nn.ReLU(),

                    # L7
                    # 256 canali in input (il livello precedente ha 256 canali di output), 1 in output
                    nn.Conv2d(in_channels=256, out_channels=1, kernel_size=5, padding=2),
                    nn.BatchNorm2d(num_features=1),
                    nn.ReLU())
        self.losses = []

    def forward(self, x):
        x = self.hidden_layers(x)
        return x


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def animation_(self):
        plt.cla()
        plt.plot(np.array(self.losses))


    def train_model(self, train_loader, lr=0.01, epochs=10):#, momentum=0.99
        optimizer = Adam(self.parameters(),lr=lr)
        criterion = nn.MSELoss()
        for e in range(epochs):
            print('epoch:', e)
            # Sets the module in training mode
            self.train()
            with torch.set_grad_enabled(True):
                for _, batch in enumerate(train_loader):
                    x = batch['rgb'].to(self.device)
                    y = batch['pbl'].to(self.device).float()
                    x = torch.reshape(x, (x.shape[0],x.shape[3],x.shape[2],x.shape[1])).float()
                    output = self(x)
                    output = torch.squeeze(output)
                    #calcolo MSE tra f(x) e pbl
                    l = criterion(output, y)

                    # backpropagation algorithm
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # acc = accuracy_score(y.to('cpu'),output.to('cpu').max(1)[1])
                    # n = batch[0].shape[0]

                    # print('accuracy at epoch', e+1, ':', acc*n )
                    # print('mse:', l.item())

                    self.losses.append(l.item())
                    # ani = FuncAnimation(plt.gcf(), self.animation_())
                    # plt.show()

            check_existing_folder('data/RGB2Gray/checkpoints/')
            torch.save(self.state_dict(), 'data/RGB2Gray/checkpoints/%s.pth'%('RGB2Gray'))
        plt.plot(self.losses)
        plt.show()

    def batch_test(self, loader):
        output_dict = {'rgb' : [], 'pbl' : []}
        for _, batch in enumerate(loader):
            x = batch['rgb']
            x = torch.reshape(x, (x.shape[0],x.shape[3],x.shape[2],x.shape[1])).float()
            x = self(x)
            output_dict['rgb'].append(x)
            output_dict['pbl'].append(batch['pbl'])
        return output_dict