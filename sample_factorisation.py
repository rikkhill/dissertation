# Run factorisation method n times and record error

import numpy as np
import pandas as pd
from performance.utils import *
from pymf import *

ratings = pd.read_csv("./data/1M/partitioned_10pc.csv")

# Zero out test set data in training set
training = ratings.copy()

training["rating"][training["training"] == 0] = 0
training = training.pivot(index="movieId", columns="userId", values="rating")
training.fillna(0, inplace=True)
training = training.as_matrix()
# For BNMF
#training[training > 0] = 1

weight_matrix = np.ones(training.shape)

# Zero out training set data in test set
test = ratings.copy()

test["rating"][test["training"] == 1] = 0
test = test.pivot(index="movieId", columns="userId", values="rating")
test.fillna(0, inplace=True)
test = test.as_matrix()
# For BNMF
#test[test > 0] = 1

# Augment movie matrix with movie age
base_movies = ratings["movieId"].unique().tolist()
movie_years = pd.read_csv("./data/1M/movie_years.csv")
# strip non-relevant entries
movie_years = movie_years[movie_years["movieId"].isin(base_movies)]
movie_years["year"] = movie_years["year"].apply(lambda x: 2000 - x)
"""
# Load user ages
base_users = ratings["userId"].unique().tolist()
user_gender = pd.read_csv("./data/1M/users.dat",
                          sep="::",
                          engine="python",
                          header=None)

user_gender.columns = ["userId", "gender", "age", "occupation", "zip"]
user_gender = user_gender[["userId", "gender"]]
user_gender = user_gender[user_gender["userId"].isin(base_users)]

user_gender["M"] = (user_gender["gender"] == "M").astype(int)
user_gender["F"] = (user_gender["gender"] == "F").astype(int)

# augments = movie_years["year"].as_matrix()
# augments = augment_vector(augments)
augments = user_gender[["M", "F"]].as_matrix()
"""

def sample_run(label, n, k, niter):
    for i in range(0, n):
        print("\tBeginning sample %d" % i)
        # nmf_model = WNMF(training, weight_matrix, num_bases=k, mask_zeros=True)
        # nmf_model = BNMF(training, num_bases=k)
        nmf_model = PMF(training, num_bases=k)
        # nmf_model = AWNMF(training.T, weight_matrix.T, augments, num_bases=k, mask_zeros=True)
        # nmf_model = ABNMF(training.T, augments, num_bases=k)
        nmf_model.factorize(niter=niter, show_progress=False)
        approx = np.dot(nmf_model.Ew.T, nmf_model.Eh.T)
        train_error = scaled_f_norm(approx, training, scaled=False)
        test_error = scaled_f_norm(approx, test, scaled=False)
        write_result("%sK%dtrain" % (label, k), train_error)
        write_result("%sK%dtest" % (label, k), test_error)
        print("\tTrain error: %f" % train_error)
        print("\tTest error: %f" % test_error)

for j in [5, 10, 30, 60, 100]:
    print("Running for K = %d" % j)
    sample_run("spmf", 4, j, 2)