cd experiments

# u++ features selected by lgbm importance
python runner.py --fe confings/fe003.yml
python runner.py --run confings/run022.yml
python select_features.py --n 300

# lgbm with lain features and pseudo labeling
python runner.py --run confings/run034.yml
# lgbm with u++ features and pseudo labeling
python runner.py --run confings/run035.yml
# lgbm with lain features and pseudo labeling 2nd
python runner.py --run confings/run039.yml

# weighted averaging by Nelder-Mead
python weighted_averaging.py

cd ../
