from numpy.core.fromnumeric import put
from torch._C import layout
import torch.nn as nn
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss
from torch.serialization import load
from torch.utils.data.dataloader import DataLoader
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from sklearn.preprocessing import normalize
from torchnet.meter import AverageValueMeter
from sklearn.metrics import accuracy_score
import utilities
import numpy as np
from torch.utils.data import DataLoader as DataLoader

class ElementVSAll(nn.Module):
    def __init__(self, input_size, n_features=128):
        super(ElementVSAll, self).__init__()

        self.in_size = input_size
        self.hidden = n_features

        # feature extractor compresses the input into the feature space
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, int(input_size/2)),
            nn.ReLU(),
            nn.Linear(int(input_size/2), int(input_size/4)),
            nn.ReLU(),
            nn.Linear(int(input_size/4), int(input_size/8)),
            nn.ReLU(),
            nn.Linear(int(input_size/8), self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1)
        )
        # random initialization of weights
        for layer in self.feature_extractor:
            layername = layer.__class__.__name__
            if layername == 'Linear':
                torch.nn.init.normal_(layer.weight, mean=0, std=0.1)

        self.classifier = nn.Sigmoid()
    
    def forward(self, data):
        pred = self.classifier(self.feature_extractor(data)) # pred is a tensor in the range [0,1].
        return torch.cat((1- pred, pred), dim=1)

# define the device available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_set, test_set, batch_size, lr=0.003, epochs=5, momentum=0.99, model_filename='model.p'):
    # plotter
    loss_logger = VisdomPlotLogger(plot_type='line', env='pav_classifier', opts={'title': 'Loss'+model_filename, 'legend':['train','test']})
    acc_logger = VisdomPlotLogger('line', env='pav_classifier', opts={'title': 'Accuracy'+model_filename,'legend':['train','test']})
    # meters
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    visdom_saver = VisdomSaver(envs=['pav_classifier'])

    # optimizer
    optim = torch.optim.SGD(model.parameters(), lr, momentum=momentum) # weight_decay = regularization!

    # loss needed to be instantiated, otherwhise will be raised the RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    criterion = BCEWithLogitsLoss()

    # normalization (1.)
    # train_set = normalize(train_set, axis=1, norm='max')
    # test_set = normalize(test_set, axis=1, norm='max')

    # obtain loaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    for e in range(epochs):
        for mode in ['train', 'test']:
            loss_meter.reset()
            print('epoch:', e, end='\r')

            loader = {
                'train' : train_dataloader,
                'test' : test_dataloader
                }
            # feed the model with train/test set to train properly.
            model.train() if mode == 'train' else model.eval()
            with torch.set_grad_enabled(mode=='train'):
                for i, batch_ in enumerate(loader[mode]):
                    x = batch_['row'].to(device)
                    y = batch_['class'].to(device)

                    output = model(x) # (4.)
                    # if output.ndim == 2:
                    #     output = torch.squeeze(output, dim=1)
                    # y = y.long() -> y.as_type(output) because otherwise RuntimeError: result type Double can't be cast to the desired output type Long will raise
                    l = criterion(output, y.type_as(output)) # (5.)
                    if mode == 'train':
                        l.backward() # (6.)
                        optim.step()
                        optim.zero_grad()

                    output = torch.where(output > 0.5, 1, 0)

                    # compute accuracy score
                    acc = accuracy_score(y.to('cpu').detach().numpy(), output.to('cpu').detach().numpy())

                    # print(y, output)
                    n = batch_['row'].shape[0]
                    loss_meter.add(l.item() * n, n)
                    acc_meter.add(acc * n, n)

                    if mode == 'train':
                        loss_logger.log(e+(i+1)/len(train_dataloader), l.item(), name='train')
            loss_logger.log(e+1, loss_meter.value()[0], name=mode)
            acc_logger.log(e+1, acc_meter.value()[0], name=mode)

        # save the model
        path = './data/ElementVSAll/'
        utilities.check_existing_folder(path=path[:-1])
        torch.save(model, path + model_filename)
        visdom_saver.save()
    return model

# def test_classifier(model, loader):
#     model.to(device)
#     predictions, labels = [], []
#     for batch in loader:
#         x = batch[0].to(device)
#         y = batch[1].to(device)
#     output = model(x)
#     preds = output.to('cpu').max(1)[1].numpy()
#     labs = y.to('cpu').numpy()
#     predictions.extend(list(preds))
#     labels.extend(list(labs))
#     return np.array(predictions), np.array(labels)