from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SimpleDataset(Dataset):
    # single channel
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.is_train = False if self.labels is None else True
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = Image.open(str(self.paths[i]))
        tfms = transforms.Compose([
            transforms.Resize(90),
            transforms.CenterCrop(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tfms(x)
        x = x.float()
        return (x, torch.tensor(self.labels[i]).float(), str(self.paths[i])) if self.is_train else (x, str(self.paths[i]))
