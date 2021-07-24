import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss, MSELoss
from torch.utils.data.dataloader import DataLoader
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from sklearn.preprocessing import normalize
from torchnet.meter import AverageValueMeter
from sklearn.metrics import accuracy_score
import utilities
from torch.utils.data import DataLoader as DataLoader, sampler
import numpy as np
from torch.optim import Adam

class ChemElementRegressor(nn.Module):
    def __init__(self, input_size, n_features=128):
        super(ChemElementRegressor, self).__init__()

        self.in_size = input_size
        self.hidden = n_features

        # feature extractor compresses the input into the feature space
        #first layer
        self.linear0 = nn.Linear(input_size, int(input_size/2)),
        self.norm0 = nn.BatchNorm1d(num_features=int(input_size/2)),
        self.relu = nn.ReLU(),

        # second layer
        self.linear1 = nn.Linear(int(input_size/2), int(input_size/4)),
        self.norm1 = nn.BatchNorm1d(num_features=int(input_size/4)),
        
        # third layer
        self.linear2 = nn.Linear(int(input_size/4), int(input_size/8)),
        self.norm2 = nn.BatchNorm1d(num_features=int(input_size/8)),
        
        # fourth layer
        self.linear3 = nn.Linear(int(input_size/8), self.hidden),
        self.norm3 = nn.ReLU(),
        
        # last layer
        self.output = nn.Linear(self.hidden, 1)
        
        self.feature_extractor = nn.Sequential(
            #first layer
            nn.Linear(input_size, int(input_size/2)),
            nn.BatchNorm1d(num_features=int(input_size/2)),
            nn.ReLU(),

            # second layer
            nn.Linear(int(input_size/2), int(input_size/4)),
            nn.BatchNorm1d(num_features=int(input_size/4)),
            nn.ReLU(),
            
            # third layer
            nn.Linear(int(input_size/4), int(input_size/8)),
            nn.BatchNorm1d(num_features=int(input_size/8)),
            nn.ReLU(),
            
            # fourth layer
            nn.Linear(int(input_size/8), self.hidden),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.hidden),
            
            # last layer
            nn.Linear(self.hidden, 1)
        )
        # random initialization of weights
        for layer in self.feature_extractor:
            layername = layer.__class__.__name__
            if layername == 'Linear':
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, data):
        # return self.feature_extractor(data)
        output0 = self.linear0(data)
        output0 = self.norm0(output0)
        output0 = self.relu(output0)

        output1 = self.linear0(output0)
        output1 = self.norm0(output1)
        output1 = self.relu(output1)
        
        output2 = self.linear0(output1)
        output2 = self.norm0(output2)
        output2 = self.relu(output2)

        output3 = self.linear0(output2)
        output3 = self.norm0(output3)
        output3 = self.relu(output3)

        output = self.output(output3)

        return output
        


class ChemElementRegressor_Convbased(nn.Module):
    def __init__(self, input_size, kernel_size=5, n_features=128):
        super(ChemElementRegressor_Convbased, self).__init__()

        self.in_size = input_size
        self.hidden = n_features
        self.kernel_size = kernel_size

        # NETWORK DEFINITION
        # first layer
        self.conv_0 = nn.Conv1d(in_channels=1, out_channels=25, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_0 = nn.MaxPool1d(kernel_size=4, stride=4)
        # self.batch_norm_0 = nn.BatchNorm1d(num_features=256)

        self.relu = nn.ReLU()

        # second layer
        self.conv_1 = nn.Conv1d(in_channels=25, out_channels=12, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_1 = nn.MaxPool1d(kernel_size=4, stride=4)
        # self.batch_norm_1 = nn.BatchNorm1d(num_features=int(input_size/4))
        
        # third layer
        self.conv_2 = nn.Conv1d(in_channels=12, out_channels=6, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        self.maxpool_2 = nn.MaxPool1d(kernel_size=2, stride=2) # output size: 64
        # self.batch_norm_2 = nn.BatchNorm1d(num_features=int(input_size/8))
        
        # fourth layer
        self.conv_3 = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=self.kernel_size, padding=int(kernel_size/2))
        # self.maxpool_3 = nn.MaxPool1d(kernel_size=3, stride=3)
        # self.batch_norm_3 = nn.BatchNorm1d(num_features=self.hidden)
        
        # last layer
        self.fc = nn.Linear(64, 1)

        self.feature_extractor = nn.Sequential(
            #first layer
            self.conv_0,
            self.maxpool_0,
            # self.batch_norm_0,
            self.relu,
            
            # second layer,
            self.conv_1,
            self.maxpool_1,
            # self.batch_norm_1,
            self.relu,
            
            # third layer,
            self.conv_2,
            self.maxpool_2,
            # self.batch_norm_2,
            self.relu,
            
            # fourth layer,
            self.conv_3,
            # self.maxpool_3,
            # self.batch_norm_3,
            self.relu,
            
            # last layer
            self.fc
        )
        # random initialization of weights
        for layer in self.feature_extractor:
            layername = layer.__class__.__name__
            if layername == 'Conv1d':
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, data):
        output0 = self.conv_0(data)
        output0 = self.maxpool_0(output0)
        output0 = self.relu(output0)

        output1 = self.conv_1(output0)
        output1 = self.maxpool_1(output1)
        output1 = self.relu(output1)

        output2 = self.conv_2(output1)
        output2 = self.maxpool_2(output2)
        output2 = self.relu(output2)

        output3 = self.conv_3(output2)
        output3 = self.relu(output3)
        
        output = self.fc(output3)

        return output

# define the device available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_regressor(model, train_set, test_set, batch_size, sampler, lr=0.003, epochs=5, momentum=0.99, model_filename='regressor.p'):
    # plotter
    loss_logger = VisdomPlotLogger(plot_type='line', env='element_regressor', opts={'title': model_filename, 'legend':['train','test']})

    # meters
    loss_meter = AverageValueMeter()
    
    # saver
    visdom_saver = VisdomSaver(envs=['element_regressor'])

    # optimizer
    # optim = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=0.2)
    optim = Adam(model.parameters(), lr, weight_decay=0.2)

    # loss needed to be instantiated, otherwhise will be raised the RuntimeError: Boolean value of Tensor with more than one value is ambiguous
    criterion = nn.MSELoss()

    # normalization (1.)
    # train_set = normalize(train_set, axis=1, norm='max')
    # test_set = normalize(test_set, axis=1, norm='max')

    # obtain loaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler['training'])
    test_dataloader = DataLoader(test_set, batch_size=batch_size, sampler=sampler['test'])
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
                # print('\n\n################################## DEBUG: DETERMINO IL BATCH #################################\n')
                for i, batch_ in enumerate(loader[mode]):
                    # print('mode=', mode, i)
                    # print('x:', batch_['row'])
                    x = batch_['row'].to(device) # index of the row

                    # print('y:', batch_['counts'])
                    y = batch_['counts'].to(device) # the value to predict
                    
                    # x = torch.unsqueeze(x,1)
                    y = torch.unsqueeze(y,1)
                    output = model(x) # (4.)

                    # y = y.long() -> y.as_type(output) because otherwise RuntimeError: result type Double can't be cast to the desired output type Long will raise
                    l = torch.sqrt(criterion(output, y.type_as(output))) # (5.)
                    if mode == 'train':
                        l.backward() # (6.)
                        optim.step()
                        optim.zero_grad()
                    n = batch_['row'].shape[0]
                    loss_meter.add(l.item() * n, n)

                    if mode == 'train':
                        loss_logger.log(e+(i+1)/len(train_dataloader), l.item(), name='train')
            
                    # print('\n\n!!!!!!!!!!!!!!!!!! DEBUG: fine batch !!!!!!!!!!!!!!!!!!')
            loss_logger.log(e+1, loss_meter.value()[0], name=mode)

        # save the model
        path = './data/ChemElementRegressor/DOggionoGiulia'
        utilities.check_existing_folder(path=path[:-1])
        torch.save(model, path + model_filename)
        visdom_saver.save()
    return model