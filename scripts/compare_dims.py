import pandas as pd
import numpy as np

full = pd.read_csv("./data/1M/ratings.dat", sep='::', engine='python')
full.columns = ['userId', 'movieId', 'rating', 'timestamp']

full = full.pivot(index="movieId", columns="userId", values="rating")
full.fillna(0, inplace=True)
full = full.as_matrix()

partitioned = pd.read_csv("./data/1M/partitioned.csv")

# Create training set with zeroed-out values for test elements
train = partitioned.copy()
train["rating"][train["training"] == 0] = 0

train = train.pivot(index="movieId", columns="userId", values="rating").fillna(0)
train.fillna(0, inplace=True)
train = train.as_matrix()

# Create test set with zeroed-out values for training elements
test = partitioned.copy()
test["rating"][test["training"] == 1] = 0

test = test.pivot(index="movieId", columns="userId", values="rating").fillna(0)
test.fillna(0, inplace=True)
test = test.as_matrix()

full_mask = full[full > 0].astype(int)
print np.sum(full_mask)

train_mask = train[train > 0].astype(int)
print np.sum(train_mask)

test_mask = test[test > 0].astype(int)
print np.sum(test_mask)

print np.sum(train_mask) + np.sum(test_mask)
