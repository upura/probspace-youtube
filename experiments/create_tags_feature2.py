from tqdm import tqdm
import pandas as pd

from ayniy.utils import Data


if __name__ == '__main__':
    train = pd.read_csv('../input/train_data.csv')
    test = pd.read_csv('../input/test_data.csv')
    freq = pd.read_csv('../input/freq.csv')
    freq = freq.sort_values('abs', ascending=False)

    tag_train_num = train['tags'].str.count('|') + 1
    tag_test_num = test['tags'].str.count('|') + 1

    for t in tqdm(freq['tag'][:50]):
        train[f'tag_contains_{t}'] = train['tags'].str.contains(t, na=False).astype(int)
        test[f'tag_contains_{t}'] = test['tags'].str.contains(t, na=False).astype(int)

    train = train.iloc[:, train.columns.str.contains('tag_contains')]
    test = test.iloc[:, test.columns.str.contains('tag_contains')]

    print(train.columns)
    """
    Index(['tag_contains_Hamamatsu', 'tag_contains_natuur',
       'tag_contains_landschap', 'tag_contains_official music video',
       'tag_contains_landscape', 'tag_contains_vevo', 'tag_contains_Hayashi',
       'tag_contains_stockshot', 'tag_contains_official video',
       'tag_contains_children songs', 'tag_contains_kids songs',
       'tag_contains_kindergarten', 'tag_contains_rhymes',
       'tag_contains_preschool', 'tag_contains_nursery rhymes',
       'tag_contains_karaoke', 'tag_contains_download', 'tag_contains_archive',
       'tag_contains_music video', 'tag_contains_baby songs',
       'tag_contains_toddlers', 'tag_contains_kids videos',
       'tag_contains_architecture', 'tag_contains_Sony',
       'tag_contains_Alternative', 'tag_contains_Rock Music',
       'tag_contains_instrumental', 'tag_contains_Super Simple Songs',
       'tag_contains_single', 'tag_contains_remix', 'tag_contains_Hiroshi',
       'tag_contains_audio', 'tag_contains_ワンオク', 'tag_contains_for kids',
       'tag_contains_food porn', 'tag_contains_hikakingames',
       'tag_contains_ヒカキンゲーム', 'tag_contains_ヒカキンゲームズ',
       'tag_contains_HoneyWorks', 'tag_contains_album', 'tag_contains_Talking',
       'tag_contains_bartender', 'tag_contains_マイクラ', 'tag_contains_かんなあきら',
       'tag_contains_Pop', 'tag_contains_Rhett and Link',
       'tag_contains_complex media', 'tag_contains_Rhett',
       'tag_contains_cocktail', 'tag_contains_RhettandLink2'],
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

    fe_name = 'fe001_top500_tag2'
    Data.dump(train_tag, f'../input/X_train_{fe_name}.pkl')
    Data.dump(test_tag, f'../input/X_test_{fe_name}.pkl')
