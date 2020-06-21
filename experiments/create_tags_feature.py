import collections
import itertools

import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    train = pd.read_csv('../input/train_data.csv')
    test = pd.read_csv('../input/test_data.csv')

    tag_train_num = train['tags'].str.count('|') + 1
    tag_test_num = test['tags'].str.count('|') + 1

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
        train[f'tag_contains_{t}'] = train['tags'].str.contains(t, na=False).astype(int)
        test[f'tag_contains_{t}'] = test['tags'].str.contains(t, na=False).astype(int)

    train = train.iloc[:, train.columns.str.contains('tag_contains')]
    test = test.iloc[:, test.columns.str.contains('tag_contains')]

    print(train.columns)
    """
    Index(['tag_contains_[none]', 'tag_contains_music', 'tag_contains_アニメ',
       'tag_contains_japan', 'tag_contains_PV', 'tag_contains_MV',
       'tag_contains_funny', 'tag_contains_live', 'tag_contains_映画',
       'tag_contains_video', 'tag_contains_Japan', 'tag_contains_ライブ',
       'tag_contains_official', 'tag_contains_the', 'tag_contains_animation',
       'tag_contains_comedy', 'tag_contains_ゲーム', 'tag_contains_新曲',
       'tag_contains_japanese', 'tag_contains_CM', 'tag_contains_youtube',
       'tag_contains_lyrics', 'tag_contains_of', 'tag_contains_pv',
       'tag_contains_ミュージック', 'tag_contains_Music', 'tag_contains_kids',
       'tag_contains_universal', 'tag_contains_ジャパン', 'tag_contains_guitar',
       'tag_contains_ユニバーサル', 'tag_contains_Japanese', 'tag_contains_anime',
       'tag_contains_mv', 'tag_contains_J-POP', 'tag_contains_rock',
       'tag_contains_動画', 'tag_contains_アイドル', 'tag_contains_高画質',
       'tag_contains_日本', 'tag_contains_歌詞', 'tag_contains_movie',
       'tag_contains_ようつべ', 'tag_contains_food', 'tag_contains_game',
       'tag_contains_cover', 'tag_contains_cooking', 'tag_contains_The',
       'tag_contains_ロック', 'tag_contains_音楽'],
      dtype='object')
    """
    train.columns = [f'tag_{i}' for i in range(len(train.columns))]
    test.columns = [f'tag_{i}' for i in range(len(train.columns))]

    train['tag_num'] = tag_train_num.fillna(0)
    test['tag_num'] = tag_test_num.fillna(0)

    train.to_csv('../input/tag_tr.csv', index=False)
    test.to_csv('../input/tag_te.csv', index=False)

    fe001_top500_tr = Data.load('../input/X_train_fe001_top500.pkl')
    fe001_top500_te = Data.load('../input/X_test_fe001_top500.pkl')

    train_tag = pd.concat([fe001_top500_tr, train], axis=1)
    test_tag = pd.concat([fe001_top500_te, test], axis=1)

    fe_name = 'fe001_top500_tag'
    Data.dump(train_tag, f'../input/X_train_{fe_name}.pkl')
    Data.dump(test_tag, f'../input/X_test_{fe_name}.pkl')
