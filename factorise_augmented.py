# Factorise ratings matrix with augmented values

import pandas as pd
from pymf import *
import sys

try:
    sys.argv[1]
except IndexError:
    # Sensible default
    K = 42
else:
    K = int(sys.argv[1])

# Lets build a ratings matrix
df = pd.read_csv("./data/1M/ratings.dat", sep='::', engine='python')
df.columns = ['userId', 'movieId', 'rating', 'timestamp']


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


# Return stack of binary indicators for vector elements in ranges
def augment_vector(v):

    print(v)

    age_range = [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
        (30, 35),
        (35, 40),
        (40, 45),
        (45, 50),
        (50, 55),
        (55, 100) # Catch all older movies

        #(0, 10),
        #(10, 20),
        #(20, 30),
        #(30, 40),
        #(40, 50),
        #(50, 100)
    ]

    stack = []

    for (a, b) in age_range:
        stack.append(((v >= a) & (v < b)).astype(int))

    return np.stack(stack, axis=0).T


# Make augmented variables for factorisation
base_movies = df["movieId"].unique().tolist()
# Load movie years
movie_years = pd.read_csv("./data/1M/movie_years.csv")
# strip non-relevant entries
movie_years = movie_years[movie_years["movieId"].isin(base_movies)]
movie_years["year"] = movie_years["year"].apply(lambda x: 2000 - x)

augments = movie_years["year"].as_matrix()
augments = augment_vector(augments)


def callout(arg):
    print(arg.frobenius_norm(complement=True))

model = AWNMF(rMatrix, weight_matrix, augments, num_bases=K, mask_zeros=True)

# Use augmented binary matrix factorisation
#model = ABNMF(rMatrix, augments, num_bases=K)
model.factorize(niter=100, show_progress=True, epoch_hook=lambda x: callout(x))

movies = model.W
users = model.H
np.savetxt("./output/factorisations/awnmf/dimmoviesK%d.csv" % K, movies)
np.savetxt("./output/factorisations/awnmf/dimusersK%d.csv" % K, users)


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
np.savetxt("./output/factorisations/awnmf/dimrelK%d.csv" % K, basis_relevance)

