import collections
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd

# from ayniy.utils import Data


if __name__ == '__main__':
    train = pd.read_csv('../input/train_data.csv')
    test = pd.read_csv('../input/test_data.csv')

    y_train = np.log1p(train['y'])

    tags_train = train['tags'].str.split('|')
    tags_test = test['tags'].str.split('|')
    tags = tags_train + tags_test
    tags.dropna(inplace=True)
    tags = list(itertools.chain(*list(tags)))
    c = collections.Counter(tags)
    freq = c.most_common(1000)
    freq = pd.DataFrame(freq)
    freq.columns = ['tag', 'cnt']

    diffs = []
    for t in tqdm(freq['tag']):
        train[f'tag_contains_{t}'] = train['tags'].str.contains(t, na=False).astype(int)
        test[f'tag_contains_{t}'] = test['tags'].str.contains(t, na=False).astype(int)
        y_train_pos = y_train[train['tags'].str.contains(t, na=False)].mean()
        y_train_neg = y_train[~train['tags'].str.contains(t, na=False)].mean()
        diff = y_train_pos - y_train_neg
        diffs.append(diff)
    freq['diff'] = diffs
    freq['diff_abs'] = freq['diff'].abs()
    freq.to_csv('../input/freq.csv', index=False, encoding='utf-8-sig')
