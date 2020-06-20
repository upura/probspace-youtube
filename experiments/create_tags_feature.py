import collections
import itertools

import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    train = pd.read_csv('../input/train_data.csv')
    test = pd.read_csv('../input/test_data.csv')

    tags_train = train['tags'].str.split('|')
    tags_test = test['tags'].str.split('|')
    tags = tags_train + tags_test
    tags.dropna(inplace=True)
    tags = list(itertools.chain(*list(tags)))
    c = collections.Counter(tags)
    freq = c.most_common(50)
    freq = pd.DataFrame(freq)
    freq.columns = ['tag', 'cnt']

    for t in freq['tag']:
        train[f'tag_contains_{t}'] = train['tags'].str.contains(t)
        test[f'tag_contains_{t}'] = test['tags'].str.contains(t)

    train.iloc[:, train.columns.str.contains('tag_contains')].to_csv('../input/tag_tr.csv', index=False)
    test.iloc[:, test.columns.str.contains('tag_contains')].to_csv('../input/tag_te.csv', index=False)

    fe001_top500_tr = Data.load('../input/X_train_fe001_top500.pkl')
    fe001_top500_te = Data.load('../input/X_test_fe001_top500.pkl')

    train_tag = pd.concat([
        fe001_top500_tr,
        train.iloc[:, train.columns.str.contains('tag_contains')]], axis=1)
    test_tag = pd.concat([
        fe001_top500_te,
        test.iloc[:, test.columns.str.contains('tag_contains')]], axis=1)

    fe_name = 'fe001_top500_tag'
    Data.dump(train_tag, f'../input/X_train_{fe_name}.pkl')
    Data.dump(test_tag, f'../input/X_test_{fe_name}.pkl')
