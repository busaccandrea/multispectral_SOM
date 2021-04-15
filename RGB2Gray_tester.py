from glob import glob
from matplotlib import pyplot as plt
from CustomDataset import CustomDataset, ToTensor
from RGB2Gray import RGB2Gray
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import os
import random
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CustomDataset('data/RGB2Gray/Andrea/PaoloMolaro/dataset.csv', root_dir='data/RGB2Gray/Andrea/PaoloMolaro/')
dataset = dataset.get_subset([24,26,25])
loader = DataLoader(dataset=dataset)

model = RGB2Gray()
model.load_state_dict(torch.load('data/RGB2Gray/checkpoints/RGB2Gray.pth'))

# Sets the module in evaluation mode.
model.eval()

# test_images = glob('data/RGB2Gray/Andrea/PaoloMolaro/PbLPatches/*.png')

# img_to_test = random.choice(test_images)

# sample = dataset.__getitem__(50)['rgb']
# sample = torch.from_numpy(sample).to(device)
# sample = torch.reshape(sample, (sample.shape[0],sample.shape[3],sample.shape[2],sample.shape[1])).float()
# sample = torch.squeeze(sample)

output_dict = model.batch_test(loader)

for output, y in zip(output_dict['rgb'],output_dict['pbl']):
    output = torch.squeeze(output)
    output = torch.detach(output).numpy()
    y = torch.squeeze(y)
    y = torch.detach(y).numpy()
    img = Image.fromarray(output)
    img.show()
    img = Image.fromarray(y)
    img.show()