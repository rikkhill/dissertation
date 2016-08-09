import numpy as np
import pandas as pd

import nimfa


K = 10

# Read in ratings data
ratings = pd.read_csv("./data/1M/partitioned_10pc.csv")

# Pivot into matrix
rm = ratings.pivot(index="movieId", columns="userId", values="rating")
rm.fillna(0, inplace=True)
rm = rm.as_matrix()

# We want a binary matrix
#rm[rm > 0] = 1

#model = nimfa.Nmf(rm, seed="nndsvd", rank=K, max_iter=100, lambda_w=1.1, lambda_h=1.1)
model = nimfa.Nmf(rm, seed="nndsvd", rank=K, max_iter=100)
model_fit = model()

movies = model_fit.basis()
users = model_fit.coef()


np.savetxt("./artefacts/dimmoviesK%d.csv" % K, movies)
np.savetxt("./artefacts/dimusersK%d.csv" % K, users)