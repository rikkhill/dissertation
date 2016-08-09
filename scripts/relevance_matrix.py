# Take hadoop MapReduce factorisations and produce relevance matrices for them

import pandas as pd
import numpy as np
"""
w = pd.read_csv("artefacts/w_advance.txt", header=None)
h = pd.read_csv("artefacts/h_advance.txt", header=None)

w_matrix = w.as_matrix()
h_matrix = h.as_matrix().T

print w_matrix.shape
print h_matrix.shape
"""

factor_dir = "./artefacts/"

K = 10

w_matrix = np.loadtxt("%sdimmoviesK%d.csv" % (factor_dir, K), delimiter=" ")
h_matrix = np.loadtxt("%sdimusersK%d.csv" % (factor_dir, K), delimiter=" ")


assert K == h_matrix.shape[0], "Dimensions must match"

np.savetxt("./output/factorisations/nimfa_bmf/dimmoviesK%d.csv" % K, w_matrix)
np.savetxt("./output/factorisations/nimfa_bmf/dimusersK%d.csv" % K, h_matrix)

df = pd.read_csv("./data/1M/partitioned_10pc.csv")
base_movies = df['movieId'].unique().tolist()

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

basis_relevance = np.dot(w_matrix.T, relevance)
np.savetxt("./output/factorisations/nimfa_bmf/dimrelK%d.csv" % K, basis_relevance)