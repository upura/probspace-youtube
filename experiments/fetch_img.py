import requests

import pandas as pd
from tqdm import tqdm


def fetch_img_from_url(row, train=True):
    url = row['thumbnail_link']
    rid = row['id']

    if train:
        file_name = f'../input/img_tr/{rid}.jpg'
    else:
        file_name = f'../input/img_te/{rid}.jpg'

    response = requests.get(url)
    img = response.content
    with open(file_name, "wb") as f:
        f.write(img)
    return 0


if __name__ == '__main__':
    train = pd.read_csv('../input/train_data.csv')
    test = pd.read_csv('../input/test_data.csv')

    for index, row in tqdm(train.iterrows()):
        fetch_img_from_url(row, True)

    for index, row in tqdm(test.iterrows()):
        fetch_img_from_url(row, False)
