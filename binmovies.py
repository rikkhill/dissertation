from pymf import BNMF
import pandas as pd
import numpy as np

K = 100

# Read in ratings data
ratings = pd.read_csv("./data/1M/ratings.dat", sep='::', engine='python')
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']

# Pivot into matrix
rm = ratings.pivot(index="movieId", columns="userId", values="rating")
rm.fillna(0, inplace=True)
rm = rm.as_matrix()

# We want a binary matrix
rm[rm > 0] = 1

# Factorise it
bnmf_model = BNMF(rm, num_bases=K)
bnmf_model.factorize(niter=66, show_progress=True)

movies = bnmf_model.W
users = bnmf_model.H
np.savetxt("./output/factorisations/bnmf/dimmoviesK%d.csv" % K, movies)
np.savetxt("./output/factorisations/bnmf/dimusersK%d.csv" % K, users)

base_movies = ratings["movieId"].unique().tolist()

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
np.savetxt("./output/factorisations/bnmf/dimrelK%d.csv" % K, basis_relevance)