# Script to implement maxios

import pandas as pd
import numpy as np

# low performance but exemplary of results

ratings = pd.read_csv("./data/1M/partitioned_10pc.csv")

# Get training set
training = ratings.copy()
training["rating"][training["training"] == 0] = 0

# Pivot into ratings matrix

X = training.pivot(index="movieId", columns="userId", values="rating")
X.fillna(0, inplace=True)

# make weight matrix

S = X > 0

# Number of bases
K = 10

eps_1 = 1
eps_2 = 1

W = np.random.random((X.shape[0], K)) + 10**-4
H = np.random.random((K, X.shape[1])) + 10**-4

Lam_1 = np.zeros(W.shape)
Lam_2 = np.zeros(H.shape)
