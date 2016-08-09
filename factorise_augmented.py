# Factorise ratings matrix with augmented values

import pandas as pd
from pymf import *
import sys
from performance.utils import *

try:
    sys.argv[1]
except IndexError:
    # Sensible default
    K = 9
else:
    K = int(sys.argv[1])

# Lets build a ratings matrix
# df = pd.read_csv("./data/1M/partitioned_10pc.csv", sep=':', engine='python')
df = pd.read_csv("./data/1M/partitioned_10pc.csv")
# df.columns = ['userId', 'movieId', 'rating', 'timestamp']
df["rating"][df["training"] == 0] = 0

ratings = df.pivot(index="movieId", columns="userId", values="rating")
ratings.fillna(0, inplace=True)

rMatrix = ratings.as_matrix().T

(d_m, d_n) = rMatrix.shape

# Lets make a bi-cross-validation weight matrix
hold_out_proportion = 0.05
m_indices = np.random.choice(d_m, int(d_m * hold_out_proportion), replace=False).tolist()
n_indices = np.random.choice(d_n, int(d_n * hold_out_proportion), replace=False).tolist()

weight_matrix = np.ones(rMatrix.shape)
weight_matrix[np.array(m_indices)[:, None], n_indices] = 0


# Make augmented variables for factorisation
base_movies = df["movieId"].unique().tolist()
# Load movie years
movie_years = pd.read_csv("./data/1M/movie_years.csv")
# strip non-relevant entries
movie_years = movie_years[movie_years["movieId"].isin(base_movies)]
movie_years["year"] = movie_years["year"].apply(lambda x: 2000 - x)

w_augments = movie_years["year"].as_matrix()
w_augments = augment_vector(w_augments)

# Load user genders
base_users = df["userId"].unique().tolist()
user_gender = pd.read_csv("./data/1M/users.dat",
                          sep="::",
                          engine="python",
                          header=None)

user_gender.columns = ["userId", "gender", "age", "occupation", "zip"]
#user_gender = user_gender[["userId", "gender"]]
user_gender = user_gender[user_gender["userId"].isin(base_users)]

age_groups = user_gender["age"].unique().tolist()

user_gender["M"] = (user_gender["gender"] == "M").astype(int)
user_gender["F"] = (user_gender["gender"] == "F").astype(int)

for age in age_groups:
    user_gender[age] = (user_gender["age"] == age).astype(int)

h_augments = user_gender[["M", "F"] + age_groups].as_matrix()


def callout(arg):
    print(arg.frobenius_norm(complement=True))

model = AWNMF(rMatrix,
              weight_matrix,
              h_augments,
              #h_augments,
              num_bases=K,
              mask_zeros=True)

# Use augmented binary matrix factorisation
#model = ABNMF(rMatrix, augments, num_bases=K)
model.factorize(niter=100, show_progress=True, epoch_hook=lambda x: callout(x))

users = model.W.T
movies = model.H.T
np.savetxt("./output/factorisations/agawnmf/dimmoviesK%d.csv" % K, movies)
np.savetxt("./output/factorisations/agawnmf/dimusersK%d.csv" % K, users)


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
np.savetxt("./output/factorisations/agawnmf/dimrelK%d.csv" % K, basis_relevance)
