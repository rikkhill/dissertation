# Run factorisation method n times and record error

import numpy as np
import pandas as pd
from performance.utils import *
from pymf import *
import sys

ratings = pd.read_csv("./data/1M/partitioned_10pc.csv")

# Zero out test set data in training set
training = ratings.copy()

training["rating"][training["training"] == 0] = 0
training = training.pivot(index="movieId", columns="userId", values="rating")
training.fillna(0, inplace=True)
training = training.as_matrix()
# For BNMF
#training[training > 0] = 1

# Zero out training set data in test set
test = ratings.copy()

test["rating"][test["training"] == 1] = 0
test = test.pivot(index="movieId", columns="userId", values="rating")
test.fillna(0, inplace=True)
test = test.as_matrix()
# For BNMF
#test[test > 0] = 1

# Make augmented variables for factorisation
base_movies = ratings["movieId"].unique().tolist()
# Load movie years
movie_years = pd.read_csv("./data/1M/movie_years.csv")
# strip non-relevant entries
movie_years = movie_years[movie_years["movieId"].isin(base_movies)]
movie_years["year"] = movie_years["year"].apply(lambda x: 2000 - x)

w_augments = movie_years["year"].as_matrix()
w_augments = augment_vector(w_augments)

movie_genres = pd.read_csv("./data/1M/movie_genres.csv")
movie_genres = movie_genres[movie_genres["movieId"].isin(base_movies)]

genre_list = movie_genres.columns[3:]

w_augments = np.concatenate((w_augments, movie_genres[genre_list].as_matrix()), axis=1)

# Load user genders
base_users = ratings["userId"].unique().tolist()
user_gender = pd.read_csv("./data/1M/users.dat",
                          sep="::",
                          engine="python",
                          header=None)

user_gender.columns = ["userId", "gender", "age", "occupation", "zip"]
user_gender = user_gender[user_gender["userId"].isin(base_users)]

age_groups = sorted(user_gender["age"].unique().tolist())

user_gender["M"] = (user_gender["gender"] == "M").astype(int)
user_gender["F"] = (user_gender["gender"] == "F").astype(int)

for age in age_groups:
    user_gender[age] = (user_gender["age"] == age).astype(int)

h_augments = user_gender[["M", "F"] + age_groups].as_matrix()

weight_matrix = np.ones(training.shape)

print training.shape
print w_augments.shape
print test.shape

training = np.concatenate((w_augments, training), axis=1)
weight_matrix = np.concatenate((np.ones(w_augments.shape), weight_matrix), axis=1)

test = np.concatenate((w_augments, test), axis=1)


def sample_run(label, n, k, niter):
    for i in range(0, n):
        print("\tBeginning sample %d" % i)
        # nmf_model = WNMF(training, weight_matrix, num_bases=k, mask_zeros=True)
        #nmf_model = BNMF(training, num_bases=k)
        # nmf_model = PMF(training, num_bases=k, augments=augments)
        nmf_model = AWNMF(training, weight_matrix, w_augments, num_bases=k, mask_zeros=True)
        #nmf_model = ABNMF(training, h_augments, num_bases=k)
        nmf_model.factorize(niter=niter, show_progress=False)
        approx = np.dot(nmf_model.W, nmf_model.H)
        train_error = scaled_f_norm(approx, training, scaled=False, augments=24)
        test_error = scaled_f_norm(approx, test, scaled=False, augments=24)
        write_result("%sK%dtrain" % (label, k), train_error)
        write_result("%sK%dtest" % (label, k), test_error)
        print("\tTrain error: %f" % train_error)
        print("\tTest error: %f" % test_error)

for j in [29, 34, 44, 54, 74, 124]:
    print("Running for K = %d" % j)
    sample_run("wnmf_final_side_info_genre_release_year", 4, j, 100)