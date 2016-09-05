# Factorise ratings matrix

import pandas as pd
from pymf import *
import sys

try:
    sys.argv[1]
except IndexError:
    # Sensible default
    K = 12
else:
    K = int(sys.argv[1])

# Lets build a ratings matrix
# df = pd.read_csv("./data/1M/ratings.dat", sep='::', engine='python')
# df.columns = ['userId', 'movieId', 'rating', 'timestamp']

df = pd.read_csv("./data/1M/partitioned_10pc.csv")
# Zero out test set elements
df["rating"][df["training"] == 0] = 0

rated_movies = df['movieId'].unique().tolist()

ratings = df.pivot(index="movieId", columns="userId", values="rating")
ratings.fillna(0, inplace=True)

rMatrix = ratings.as_matrix()

(d_m, d_n) = rMatrix.shape

# Lets make a bi-cross-validation weight matrix
hold_out_proportion = 0.05
m_indices = np.random.choice(d_m, int(d_m * hold_out_proportion), replace=False).tolist()
n_indices = np.random.choice(d_n, int(d_n * hold_out_proportion), replace=False).tolist()

weight_matrix = np.ones(rMatrix.shape)
# weight_matrix[np.array(m_indices)[:, None], n_indices] = 0


def callout(arg):
    print(arg.frobenius_norm(complement=True))

nmf_model = WNMF(rMatrix, weight_matrix, num_bases=K, mask_zeros=True)
#nmf_model = BNMF(rMatrix, weight_matrix, num_bases=K, mask_zeros=True)
nmf_model.factorize(niter=100, show_progress=True, epoch_hook=lambda x: callout( x))

movies = nmf_model.W
users = nmf_model.H
np.savetxt("./output/factorisations/wnmf/dimmoviesK%d.csv" % K, movies)
np.savetxt("./output/factorisations/wnmf/dimusersK%d.csv" % K, users)

base_movies = df["movieId"].unique().tolist()

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
np.savetxt("./output/factorisations/wnmf/dimrelK%d.csv" % K, basis_relevance)

