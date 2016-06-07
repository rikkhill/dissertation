# Take ratings data and trim 20% of ratings per user into out-of-sample set

import pandas as pd
import numpy as np

# Percent of data to hold out
holdout_pc = 10

ratings = pd.read_csv("./data/1M/ratings.dat",
                     sep='::',
                     engine='python',
                     header=None)

ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_users = ratings.groupby("userId")

test_set_idx = []

for user, group in ratings_users:
    holdout = int(len(group) * (holdout_pc / 100.0))
    test_set_idx += group["timestamp"].nlargest(holdout).index.tolist()

test_set_idx = ratings.index.isin(test_set_idx)

ratings["training"] = np.array(~test_set_idx).astype(int)

ratings.to_csv('./data/1M/partitioned_%dpc.csv' % holdout_pc, index_label="Id")

