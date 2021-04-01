from CustomDataset import CustomDataset, ToTensor
from RGB2Gray import RGB2Gray
from torch.utils.data import Dataset, DataLoader


# load dataset
custom_data_set = CustomDataset(csv_file='data/RGB2Gray/Andrea/PaoloMolaro/dataset.csv', root_dir='data/RGB2Gray/Andrea/PaoloMolaro/')
dataloader = DataLoader(custom_data_set, batch_size=10, shuffle=True)

# create/load model
model = RGB2Gray()
""" ADD LOADING MODEL """

# train model section
#   choose n. of epochs
epochs = 10


# train
model.train_model(dataloader, epochs=epochs)