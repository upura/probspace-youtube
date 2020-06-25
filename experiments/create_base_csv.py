import re
import unicodedata

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


def is_japanese(string):
    count = 0
    for ch in str(string):
        try:
            name = unicodedata.name(ch)
        except:
            continue
        if "CJK UNIFIED" in name \
                or "HIRAGANA" in name \
                or "KATAKANA" in name:
            count += 1
    return count


def count_alphabet(string):
    r = re.compile(r"[a-z|A-Z]+")
    return len("".join(r.findall(str(string))))


def count_number(string):
    r = re.compile(r"[0-9]+")
    return len("".join(r.findall(str(string))))


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

    delete_cols = [
        'publishedAt',
        'collection_date',
        'collection_date_hour',
        'collection_date_minute',
        'thumbnail_link'
    ]
    train.drop(delete_cols, axis=1, inplace=True)
    test.drop(delete_cols, axis=1, inplace=True)

    train['tag_num'] = train['tags'].str.count('|') + 1
    test['tag_num'] = test['tags'].str.count('|') + 1
    train['tag_num'].fillna(0, inplace=True)
    test['tag_num'].fillna(0, inplace=True)

    train["likes_mul_dislikes"] = train['likes'] * train['dislikes']
    train["eval_count"] = train['likes'] + train['dislikes']
    train["likes_ratio"] = train['likes'] / train["eval_count"]
    train["dislikes_ratio"] = train['dislikes'] / train["eval_count"]
    train["comment_count_mul_eval_count"] = train["comment_count"] * train["eval_count"]

    test["eval_count"] = test['likes'] + test['dislikes']
    test["likes_ratio"] = test['likes'] / test["eval_count"]
    test["dislikes_ratio"] = test['dislikes'] / test["eval_count"]
    test["comment_count_mul_eval_count"] = test["comment_count"] * test["eval_count"]

    # 日本語を含むかかどうかの判定
    train["title_ja_count"] = train.title.apply(is_japanese)
    test["title_ja_count"] = test.title.apply(is_japanese)
    train["channelTitle_ja_count"] = train.channelTitle.apply(is_japanese)
    test["channelTitle_ja_count"] = test.channelTitle.apply(is_japanese)
    train["description_ja_count"] = train.description.apply(is_japanese)
    test["description_ja_count"] = test.description.apply(is_japanese)

    train["title_ja_ratio"] = train.title_ja_count / train.title.apply(lambda x: len(str(x)))
    test["title_ja_ratio"] = test.title_ja_count / test.title.apply(lambda x: len(str(x)))
    train["channelTitle_ja_ratio"] = train.channelTitle_ja_count / train.channelTitle.apply(lambda x: len(str(x)))
    test["channelTitle_ja_ratio"] = test.channelTitle_ja_count / test.channelTitle.apply(lambda x: len(str(x)))
    train["description_ja_ratio"] = train.title_ja_count / train.description.apply(lambda x: len(str(x)))
    test["description_ja_ratio"] = test.title_ja_count / test.description.apply(lambda x: len(str(x)))

    # アルファベットのカウント
    train["title_en_count"] = train.title.apply(count_alphabet)
    test["title_en_count"] = test.title.apply(count_alphabet)
    train["channelTitle_en_count"] = train.channelTitle.apply(count_alphabet)
    test["channelTitle_en_count"] = test.channelTitle.apply(count_alphabet)
    train["description_en_count"] = train.description.apply(count_alphabet)
    test["description_en_count"] = test.description.apply(count_alphabet)

    train["title_en_ratio"] = train.title_en_count / train.title.apply(lambda x: len(str(x)))
    test["title_en_ratio"] = test.title_en_count / test.title.apply(lambda x: len(str(x)))
    train["channelTitle_en_ratio"] = train.channelTitle_en_count / train.channelTitle.apply(lambda x: len(str(x)))
    test["channelTitle_en_ratio"] = test.channelTitle_en_count / test.title.apply(lambda x: len(str(x)))
    train["description_en_ratio"] = train.title_en_count / train.description.apply(lambda x: len(str(x)))
    test["description_en_ratio"] = test.title_en_count / test.description.apply(lambda x: len(str(x)))

    # 数字のカウント
    train["description_num_count"] = train.description.apply(count_number)
    test["description_num_count"] = test.description.apply(count_number)
    train["description_num_ratio"] = train.description_num_count / train.description.apply(lambda x: len(str(x)))
    test["description_num_ratio"] = test.description_num_count / test.description.apply(lambda x: len(str(x)))

    # urlのカウント
    train["description_url_count"] = train.description.apply(lambda x: str(x).count("://"))
    test["description_url_count"] = test.description.apply(lambda x: str(x).count("://"))

    # collection_date_minus_publishedAt
    train["likes_per_period"] = train.likes / train.collection_date_minus_publishedAt
    test["likes_per_period"] = test.likes / test.collection_date_minus_publishedAt
    train["dislikes_per_period"] = train.dislikes / train.collection_date_minus_publishedAt
    test["dislikes_per_period"] = test.dislikes / test.collection_date_minus_publishedAt
    train["comment_per_period"] = train.comment_count / train.collection_date_minus_publishedAt
    test["comment_per_period"] = test.comment_count / test.collection_date_minus_publishedAt

    train['y'] = np.log1p(train['y'])

    train.to_csv('../input/train_data_base.csv', index=False)
    test.to_csv('../input/test_data_base.csv', index=False)
    print(train.shape)      # (19720, 65)
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
       'collection_date_dow_sin', 'tag_num', 'likes_mul_dislikes',
       'eval_count', 'likes_ratio', 'dislikes_ratio',
       'comment_count_mul_eval_count', 'title_ja_count',
       'channelTitle_ja_count', 'description_ja_count', 'title_ja_ratio',
       'channelTitle_ja_ratio', 'description_ja_ratio', 'title_en_count',
       'channelTitle_en_count', 'description_en_count', 'title_en_ratio',
       'channelTitle_en_ratio', 'description_en_ratio',
       'description_num_count', 'description_num_ratio',
       'description_url_count', 'likes_per_period', 'dislikes_per_period',
       'comment_per_period'],
      dtype='object')
    """
