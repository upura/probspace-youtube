import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    ef_tr = pd.read_csv('../input/efficient_tr.csv')
    ef_te = pd.read_csv('../input/efficient_te.csv')

    fe001_top500_tr = Data.load('../input/X_train_fe001_top500.pkl')
    fe001_top500_te = Data.load('../input/X_test_fe001_top500.pkl')

    train_tag = pd.concat([fe001_top500_tr, ef_tr], axis=1)
    test_tag = pd.concat([fe001_top500_te, ef_te], axis=1)

    fe_name = 'fe001_top500_ef'
    Data.dump(train_tag, f'../input/X_train_{fe_name}.pkl')
    Data.dump(test_tag, f'../input/X_test_{fe_name}.pkl')
