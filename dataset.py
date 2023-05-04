# Torch imports
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

class LongTailedMNIST(Dataset):
    mode='TRAIN'
    def __init__(self):
        tf=transforms.Compose([
            transforms.ToTensor()
        ]) # mnist is already normalised 0 to 1

        trainset=MNIST(
            root="./data/train/",
            train=True,
            download=True,
            transform=tf
        )
        valset=MNIST(
            root="./data/val/",
            train=False,
            download=True,
            transform=tf
        )
        self.data={
            'TRAIN':trainset,
            'VAL':valset
        }

    def __getitem__(self,idx):
        return self.data[self.mode][idx]
    
    def __len__(self):
        return len(self.data[self.mode])
    
    def train(self):
        self.mode='TRAIN'
    
    def eval(self):
        self.mode='VAL'
