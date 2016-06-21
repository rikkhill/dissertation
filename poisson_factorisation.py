# Poisson factorisation of MovieLens data

import pandas as pd
from pymf import *

K = 30

df = pd.read_csv("./data/1M/partitioned_10pc.csv")
df["rating"][df["training"] == 0] = 0


ratings = df.pivot(index="movieId", columns="userId", values="rating")
ratings.fillna(0, inplace=True)
base_movies = df["movieId"].unique().tolist()
rMatrix = ratings.as_matrix()

# rMatrix[rMatrix > 0] = 1

model = PMF(rMatrix, num_bases=K)

model.factorize(niter=100, show_progress=True)

movies = model.Ew.T
users = model.Eh.T
approx = np.dot(movies, users)

print movies.shape
print users.shape

print approx.shape

np.savetxt("./output/factorisations/pmf/dimmoviesK%d.csv" % K, movies)
np.savetxt("./output/factorisations/pmf/dimusersK%d.csv" % K, users)

# Get the tag relevance matrix
gr = pd.read_csv("./data/genome/tag_relevance.dat", header=None, sep='\t')
gr.columns = ["movieId", "tagId", "relevance"]

# Trim all movies that aren't in the base movies
gr = gr[gr["movieId"].isin(base_movies)]


gr_movies = set(gr["movieId"].unique().tolist())
setdiff = [bm for bm in base_movies if bm not in gr_movies]
empty_data = pd.DataFrame([(m_id, 0, 0) for m_id in setdiff])
empty_data.columns = ["movieId", "tagId", "relevance"]

gr = gr.append(empty_data)

# Pad out the pivot
relevance = gr.pivot(index="movieId", columns="tagId", values="relevance")
relevance.fillna(0, inplace=True)
relevance = relevance.as_matrix()

basis_relevance = np.dot(movies.T, relevance)
np.savetxt("./output/factorisations/pmf/dimrelK%d.csv" % K, basis_relevance)