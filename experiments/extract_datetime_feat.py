import pandas as pd

from ayniy.preprocessing import datatime_parser


def change_to_Date(train, test, input_column_name, output_column_name):
    """https://prob.space/competitions/youtube-view-count/discussions/Oregin-Post72d2ce9af80ab2664da5
    """
    _train = train.copy()
    _test = test.copy()
    _train[output_column_name] = _train[input_column_name].map(lambda x: x.split('.'))
    _test[output_column_name] = _test[input_column_name].map(lambda x: x.split('.'))
    _train[output_column_name] = _train[output_column_name].map(lambda x: '20' + x[0] + '-' + x[2] + '-' + x[1] + 'T00:00:00.000Z')
    _test[output_column_name] = _test[output_column_name].map(lambda x: '20' + x[0] + '-' + x[2] + '-' + x[1] + 'T00:00:00.000Z')
    return _train, _test


if __name__ == '__main__':
    train = pd.read_csv('../input/train_data.csv')
    test = pd.read_csv('../input/test_data.csv')

    train, test = change_to_Date(train, test, 'collection_date', 'collection_date')
    train, test = datatime_parser(train, test, col_definition={'encode_col': ['publishedAt', 'collection_date']})

    train['collection_date_minus_publishedAt'] = [
        diff.seconds // 60 for diff in (pd.to_datetime(train['collection_date']) - pd.to_datetime(train['publishedAt']))
    ]
    test['collection_date_minus_publishedAt'] = [
        diff.seconds // 60 for diff in (pd.to_datetime(test['collection_date']) - pd.to_datetime(test['publishedAt']))
    ]

    train.drop(['publishedAt',
                'collection_date',
                'collection_date_hour',
                'collection_date_minute',
                'thumbnail_link'], axis=1, inplace=True)
    test.drop(['publishedAt',
               'collection_date',
               'collection_date_hour',
               'collection_date_minute',
               'thumbnail_link'], axis=1, inplace=True)

    print(train.shape)  # (19720, 25)
    train.to_csv('../input/train_data_convert.csv', index=False)
    test.to_csv('../input/test_data_convert.csv', index=False)
