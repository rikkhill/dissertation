# Factorise ratings matrix

import pandas as pd
from pymf import *
import sys

try:
    sys.argv[1]
except IndexError:
    # Sensible default
    K = 10
else:
    K = int(sys.argv[1])

# Lets build a ratings matrix
# df = pd.read_csv("./data/1M/ratings.dat", sep='::', engine='python')
# df.columns = ['userId', 'movieId', 'rating', 'timestamp']
"""
df = pd.read_csv("./data/FF/dense_10pc_partition_purchases.csv")

productmap = pd.DataFrame()
productmap["ProductId"] = df["ProductId"].sort_values().unique().copy()
productmap["ProductIndex"] = range(1, len(productmap) + 1)

usermap = pd.DataFrame()
usermap["UserID"] = df["UserID"].sort_values().unique().copy()
usermap["UserIndex"] = range(1, len(usermap) + 1)

df = pd.merge(df, productmap, on="ProductId")
df = pd.merge(df, usermap, on="UserID")
df = df[["ProductIndex", "UserIndex", "training"]].drop_duplicates()

"""
df = pd.read_csv("./data/1M/partitioned_10pc.csv")
# Zero out test set elements
df["rating"][df["training"] == 0] = 0
df["rating"][df["training"] > 0] = 1

#rated_movies = df['movieId'].unique().tolist()

#ratings = df.pivot(index="", columns="UserIndex", values="value")
ratings = df.pivot(index="movieId", columns="userId", values="rating")
ratings.fillna(0, inplace=True)

rMatrix = ratings.as_matrix()
(d_m, d_n) = rMatrix.shape

# Lets make a bi-cross-validation weight matrix
#hold_out_proportion = 0.9
#m_indices = np.random.choice(d_m, int(d_m * hold_out_proportion), replace=False).tolist()
#n_indices = np.random.choice(d_n, int(d_n * hold_out_proportion), replace=False).tolist()

#selection_matrix = np.random.binomial(1, 0.5, rMatrix.shape)
#rMatrix *= selection_matrix

weight_matrix = np.ones(rMatrix.shape)
"""

data = pd.read_csv("./data/synthetic/synth_mvnorm100_100_10")
M = data.pivot(index="artistId", columns="userId", values="value")
M.fillna(0, inplace=True)
rMatrix = M.as_matrix()
"""

for i in range(0, 20):
    print "factorising  set %d" % i
    #nmf_model = WNMF(rMatrix, weight_matrix, num_bases=K, mask_zeros=True)
    nmf_model = BNMF(rMatrix, num_bases=K)
    nmf_model.factorize(niter=85, show_progress=True)

    movies = nmf_model.W
    users = nmf_model.H
    np.savetxt("./output/factorisations/bnmf10/dimmoviesK%dI%d.csv" % (K, i), movies)
    np.savetxt("./output/factorisations/bnmf10/dimusersK%dI%d.csv" % (K, i), users)

