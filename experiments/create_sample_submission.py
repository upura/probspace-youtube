import pandas as pd


test = pd.read_csv('../input/test_data.csv')
test['y'] = 9999999999
test[['id', 'y']].to_csv('../input/sample_submission.csv', index=False)
