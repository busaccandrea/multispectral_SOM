import matplotlib.pyplot as plt
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

def split_data(data):
    np.random.shuffle(data)
    train_indices = int(data.shape[0]*0.7)
    train_data = data[0 : train_indices]
    test_data = data[train_indices :]
    return [train_data, test_data]

def calculate_loss(model, dataloader, loss_fn=nn.MSELoss(), flatten=True, conditional=False):
    losses = []
    for batch in dataloader:
    # for batch, labels in dataloader:
        batch = batch.to(device)
        # labels = labels.to(device)
        
        if flatten:
            batch = batch.view(batch.size(0), 2048)
            
        if conditional:
            loss = loss_fn(batch, model(batch))
        else:
            loss = loss_fn(batch, model(batch))
            
        losses.append(loss)

    return (sum(losses)/len(losses)).item() # calculate mean


def evaluate(losses, autoencoder, dataloader, flatten=True, vae=False, conditional=False, title=""):
#     display.clear_output(wait=True)
    if vae and conditional:
        model = lambda x, y: autoencoder(x, y)[0]
    elif vae:
        model = lambda x: autoencoder(x)[0]
    else:
        model = autoencoder

    loss = calculate_loss(model, dataloader, flatten=flatten, conditional=conditional)
    # show_visual_progress(model, test_dataloader, flatten=flatten, vae=vae, conditional=conditional, title=title)
    
    losses.append(loss)

def train(net, dataloader, test_dataloader, epochs=5, flatten=False, loss_fn=nn.MSELoss(), title=None):
    optim = torch.optim.Adam(net.parameters())
    
    train_losses = []
    validation_losses = []
    
    for i in range(epochs):
        print('epoch:', i)
        # for batch, labels in dataloader:
        for batch in dataloader:
            batch = batch.to(device)
            if flatten:
                batch = batch.view(batch.size(0), 2048)
            optim.zero_grad()
            loss = loss_fn(batch, net(batch))
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        if title:
            image_title = f'{title} - Epoch {i}'
        
        evaluate(validation_losses, net, test_dataloader, flatten, title=image_title)

    torch.save({
            'epoch': i,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, PATH)
    plt.plot(train_losses)
    plt.show()

def calculate_nparameters(model):
    def times(shape):
        parameters = 1
        for layer in list(shape):
            parameters *= layer
        return parameters
    
    layer_params = [times(x.size()) for x in list(model.parameters())]
    
    return sum(layer_params)

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden=10):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, hidden)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

data_ = sparse.load_npz('data/sparse_data.npz')
data = data_.toarray()

[train_data, test_data] = split_data(data)

training_set = torch.tensor(train_data).float()
test_set = torch.tensor(train_data).float()

train_dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
autoencoder = Autoencoder(input_size=2048, hidden=8).to(device)

train(autoencoder, train_dataloader, test_dataloader, epochs=10, flatten=True, title='Autoencoder')
torch.save({
            'epoch': epochs,
            'model_state_dict': autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
