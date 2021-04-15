import matplotlib.pyplot as plt
from numpy.core.shape_base import block
from scipy import sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import load
from torchvision import datasets, transforms
import numpy as np

class Pavencoder(nn.Module):
    def __init__(self, input_size, hidden=10):
        super(Pavencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size,hidden),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden, input_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # return losses.item()
    return (sum(losses)/len(losses)).item() # calculate mean

def train(net, train_dataloader, test_dataloader, epochs=5, flatten=False, loss_fn=nn.MSELoss(), save_state=True, verbose=True):
    """ 
        Method to train the model. 
        net: model
        train_dataloader: train dataset
        test_dataloader: test dataset
        epochs: number of epochs.
        loss_fn: loss function to use. (default: MSE)
        save_state: if true, after the last epoch the model will be saved.
     """
    optim = torch.optim.Adam(net.parameters(), weight_decay=0.01)
    
    train_losses = []
    validation_losses = []
    
    for i in range(epochs):
        print('epoch:', i, end='\r')
        # for batch, labels in train_dataloader:
        for batch in train_dataloader:
            batch = batch.to(device)
            if flatten:
                batch = batch.view(batch.size(0), 2048)
            optim.zero_grad()
            loss = loss_fn(batch, net(batch))
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        
        evaluate(validation_losses, net, test_dataloader, flatten)
        if save_state:
            torch.save(net, 'data/pavencoder/model.pt')
            torch.save({
                    'epoch': i,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss
                    }, 'data/pavencoder/model_checkpoint.pt')
    if verbose:
        plt.plot(train_losses, 'b', validation_losses, 'r')
        plt.plot(validation_losses, 'r')
        # show plot in fullscreen
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()

def calculate_nparameters(model):
    def times(shape):
        parameters = 1
        for layer in list(shape):
            parameters *= layer
        return parameters
    
    layer_params = [times(x.size()) for x in list(model.parameters())]
    
    return sum(layer_params)

def split_data(data):
    """ 
        Given a 2d-array this method splits 
     """
    np.random.shuffle(data.copy())
    train_indices = int(data.shape[0]*0.7)
    train_data = data[0 : train_indices]
    test_data = data[train_indices :]
    return [train_data, test_data]

def split_data_from_numpy(data_filename):
    """ Get tensors from numpy 2darray.
        returns: [training_set, test_set]
            training_set: tensor
            test_set: tensor
    """
    data = np.load(data_filename)
    [train_data, test_data] = split_data(data)
    np.save('train_dataset.npy', train_data)
    np.save('test_dataset.npy', test_data)
    training_set = torch.tensor(train_data).float()
    test_set = torch.tensor(test_data).float()
    
    return [training_set, test_set]

def load_model_for_inference(model_filename, checkpoint_filename):
    """ Load given model and checkpoint and inizializes dictionaries.
        Use this method to continue a training

        returns:    [model, optimizer, checkpoint]
            model: nn.model
        """
    model = torch.load(model_filename)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    """ Pytorch doc:
            Remember that you must call model.eval() to set dropout and batch normalization layers 
            to evaluation mode before running inference. Failing to do this will yield inconsistent inference results. """
    model.eval()

    return [model, optimizer, checkpoint]