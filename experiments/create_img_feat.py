from efficientnet_pytorch import EfficientNet
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Need to add path when you run in Google Colab
# import sys
# sys.path.append('/content/drive/My Drive/probspace-youtube')
from src.datasets import SimpleDataset
from src.utils import seed_everything


class EfficientNetB3(nn.Module):
    def __init__(self):
        super(EfficientNetB3, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b3")
        self.li = nn.Linear(1536, 50)

    def forward(self, x):
        bs, _, _, _ = x.shape
        x = self.model.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.li(x)
        return x


def get_features_from_loader(loader):
    train_img = []
    for batch in tqdm(loader):
        if len(batch) == 2:
            x, _ = batch
        elif len(batch) == 3:
            x, _, _ = batch
        x = x.to(device)
        features = model(x)
        train_img.append(features.detach().cpu().numpy())
    train_img = np.concatenate(train_img, axis=0)
    train_img = train_img
    return train_img


if __name__ == '__main__':

    run_name = 'nn000'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    INPUT_DIR = '../input/'
    train = pd.read_csv(INPUT_DIR + 'train_data.csv')
    test = pd.read_csv(INPUT_DIR + 'test_data.csv')

    TRAIN_IMG_PATH = '../input/img_tr/'
    TEST_IMG_PATH = '../input/img_te/'

    train_paths = [TRAIN_IMG_PATH + f'{x}.jpg' for x in train['id'].values]
    train_labels = np.log1p(train['y'])
    train_dataset = SimpleDataset(train_paths, labels=train_labels, transform=None)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=128)

    test_paths = [TEST_IMG_PATH + f'{x}.jpg' for x in test['id'].values]
    test_dataset = SimpleDataset(test_paths, labels=None, transform=None)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)

    model = EfficientNetB3()
    model = model.to(device)
    train_img = get_features_from_loader(train_loader)
    test_img = get_features_from_loader(test_loader)

    train_img = pd.DataFrame(train_img)
    test_img = pd.DataFrame(test_img)

    train_img.columns = [f'efficient_{i}' for i in range(50)]
    test_img.columns = [f'efficient_{i}' for i in range(50)]

    train_img.to_csv('../input/efficient_tr.csv', index=False)
    test_img.to_csv('../input/efficient_te.csv', index=False)
