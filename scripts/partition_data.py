# Take ratings data and trim 20% of ratings per user into out-of-sample set

import pandas as pd
import numpy as np

ratings = pd.read_csv("./data/1M/ratings.dat",
                     sep='::',
                     engine='python',
                     header=None)

ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings_users = ratings.groupby("userId")

test_set_idx = []

for user, group in ratings_users:
    twenty_pc = int(len(group) * 0.2)
    test_set_idx += group["timestamp"].nlargest(twenty_pc).index.tolist()

test_set_idx = ratings.index.isin(test_set_idx)

ratings["training"] = np.array(~test_set_idx).astype(int)

ratings.to_csv('./data/1M/partitioned.csv', index_label="Id")

