from CustomDataset import CustomDataset, ToTensor
from RGB2Gray import RGB2Gray
from torch.utils.data import Dataset, DataLoader
import torch
import os


# load dataset
custom_data_set = CustomDataset(csv_file='data/RGB2Gray/Andrea/PaoloMolaro/dataset.csv', root_dir='data/RGB2Gray/Andrea/PaoloMolaro/')
dataloader = DataLoader(custom_data_set, batch_size=5, shuffle=True)

# create/load model
model = RGB2Gray()
if os.path.exists('data/RGB2Gray/checkpoints/RGB2Gray.pth'):
    model.load_state_dict(torch.load('data/RGB2Gray/checkpoints/RGB2Gray.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

# train model section
#   choose n. of epochs
epochs = 1000


# train
model.train_model(dataloader, epochs=epochs)