# Run factorisation method n times and record error

import numpy as np
import pandas as pd
from performance.utils import *
from pymf import *
import sys

ff = pd.read_csv("./data/FF/dense_10pc_partition_purchases.csv")

productmap = pd.DataFrame()
productmap["ProductId"] = ff["ProductId"].sort_values().unique().copy()
productmap["ProductIndex"] = range(1, len(productmap) + 1)

usermap = pd.DataFrame()
usermap["UserID"] = ff["UserID"].sort_values().unique().copy()
usermap["UserIndex"] = range(1, len(usermap) + 1)

ff = pd.merge(ff, productmap, on="ProductId")
ff = pd.merge(ff, usermap, on="UserID")
ff = ff[["ProductIndex", "UserIndex", "training"]].drop_duplicates()
ff["value"] = 1

# Zero out test set data in training set
training = ff.copy()

training["value"][training["training"] == 0] = 0
training = training.pivot(index="ProductIndex", columns="UserIndex", values="value")
training.fillna(0, inplace=True)
training = training.as_matrix()
# For BNMF
training[training > 0] = 1

# Zero out training set data in test set
test = ff.copy()

test["value"][test["training"] == 1] = 0
test = test.pivot(index="ProductIndex", columns="UserIndex", values="value")
test.fillna(0, inplace=True)
test = test.as_matrix()
# For BNMF
test[test > 0] = 1

"""
# Make augmented variables for factorisation
base_movies = ratings["movieId"].unique().tolist()
# Load movie years
movie_years = pd.read_csv("./data/1M/movie_years.csv")
# strip non-relevant entries
movie_years = movie_years[movie_years["movieId"].isin(base_movies)]
movie_years["year"] = movie_years["year"].apply(lambda x: 2000 - x)

w_augments = movie_years["year"].as_matrix()
w_augments = augment_vector(w_augments)

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

h_augments = user_gender[["M", "F"]].as_matrix()

training = np.concatenate((h_augments, training), axis=1)

test = np.concatenate((h_augments, test), axis=1)
"""

base_products = pd.read_csv("./data/FF/productinfo.csv",
                            sep=",",
                            error_bad_lines=False,
                            warn_bad_lines=False)
base_products.rename(columns={'ProductID': 'ProductId'}, inplace=True)

print len( base_products["Family"].unique().tolist())

product_gender = pd.merge(productmap,
                      base_products[["ProductId", "GenderCat"]],
                      how="left",
                      on="ProductId")

h_augments = pd.DataFrame()
h_augments["M"] = (product_gender["GenderCat"] == "Men").astype(int)
h_augments["W"] = (product_gender["GenderCat"] == "Women").astype(int)
h_augments["K"] = (product_gender["GenderCat"] == "Kids").astype(int)

h_augments = h_augments.as_matrix()


training = np.concatenate((h_augments, training), axis=1)
test = np.concatenate((h_augments, test), axis=1)

weight_matrix = np.ones(training.shape)

def sample_run(label, n, k, niter):
    for i in range(0, n):
        print("\tBeginning sample %d" % i)
        # nmf_model = WNMF(training, weight_matrix, num_bases=k, mask_zeros=True)
        #nmf_model = BNMF(training, num_bases=k)
        # nmf_model = PMF(training, num_bases=k, augments=augments)
        #nmf_model = AWNMF(training, weight_matrix, h_augments, num_bases=k, mask_zeros=True)
        nmf_model = ABNMF(training, h_augments, num_bases=k)
        nmf_model.factorize(niter=niter, show_progress=False)
        approx = np.dot(nmf_model.W, nmf_model.H)
        train_error = scaled_f_norm(approx, training, scaled=False)
        test_error = scaled_f_norm(approx, test, scaled=False)
        write_result("%sK%dtrain" % (label, k), train_error)
        write_result("%sK%dtest" % (label, k), test_error)
        print("\tTrain error: %f" % train_error)
        print("\tTest error: %f" % test_error)

for j in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 65, 70, 80, 90]:
    print("Running for K = %d" % j)
    sample_run("dense_apbnmf_ff2", 4, j, 55)