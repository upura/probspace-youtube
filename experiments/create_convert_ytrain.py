import numpy as np
import pandas as pd

from ayniy.utils import Data


train = pd.read_csv('../input/train_data.csv')

y_train = train['y']
y_train = np.log1p(y_train)
Data.dump(y_train, '../input/y_train_fe000.pkl')
