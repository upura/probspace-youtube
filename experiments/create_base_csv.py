import numpy as np
import pandas as pd

from ayniy.preprocessing import datatime_parser, circle_encoding


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

    train, test = circle_encoding(train, test, col_definition={'encode_col': [
        'publishedAt_month',
        'publishedAt_day',
        'publishedAt_dow',
        'publishedAt_hour',
        'publishedAt_minute',
        'collection_date_month',
        'collection_date_day',
        'collection_date_dow'
    ]})

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

    train['y'] = np.log1p(train['y'])

    train.to_csv('../input/train_data_convert.csv', index=False)
    test.to_csv('../input/test_data_convert.csv', index=False)
    print(train.shape)      # (19720, 41)
    print(train.columns)
    """
    Index(['id', 'video_id', 'title', 'channelId', 'channelTitle', 'categoryId',
       'tags', 'likes', 'dislikes', 'comment_count', 'comments_disabled',
       'ratings_disabled', 'description', 'y', 'publishedAt_year',
       'publishedAt_month', 'publishedAt_day', 'publishedAt_dow',
       'publishedAt_hour', 'publishedAt_minute', 'collection_date_year',
       'collection_date_month', 'collection_date_day', 'collection_date_dow',
       'collection_date_minus_publishedAt', 'publishedAt_month_cos',
       'publishedAt_month_sin', 'publishedAt_day_cos', 'publishedAt_day_sin',
       'publishedAt_dow_cos', 'publishedAt_dow_sin', 'publishedAt_hour_cos',
       'publishedAt_hour_sin', 'publishedAt_minute_cos',
       'publishedAt_minute_sin', 'collection_date_month_cos',
       'collection_date_month_sin', 'collection_date_day_cos',
       'collection_date_day_sin', 'collection_date_dow_cos',
       'collection_date_dow_sin'],
      dtype='object')
    """
