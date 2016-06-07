# Run factorisation method n times and record error

import numpy as np
import pandas as pd
from performance.utils import *
from pymf import *

# Label for this run
label = "wnmf"

# Number of times to sample factorisation
n = 10

# Number of factors
K = 10

# Number of iterations to run factorisation
niter = 100

ratings = pd.read_csv("./data/1M/partitioned_10pc.csv")

# Zero out test set data in training set
training = ratings.copy()

training["rating"][training["training"] == 0] = 0
training = training.pivot(index="movieId", columns="userId", values="rating")
training.fillna(0, inplace=True)
training = training.as_matrix()
weight_matrix = np.ones(training.shape)

# Zero out training set data in test set
test = ratings.copy()

test["rating"][test["training"] == 1] = 0
test = test.pivot(index="movieId", columns="userId", values="rating")
test.fillna(0, inplace=True)
test = test.as_matrix()


for i in range(0, n):
    print("Beginning sample %d" % i)
    nmf_model = WNMF(training, weight_matrix, num_bases=K, mask_zeros=True)
    nmf_model.factorize(niter=100, show_progress=False)
    approx = np.dot(nmf_model.W, nmf_model.H)
    train_error = scaled_f_norm(approx, training, scaled=False)
    test_error = scaled_f_norm(approx, test, scaled=False)
    write_result("%sK%dtrain" % (label, K), train_error)
    write_result("%sK%dtest" % (label, K), test_error)
    print("\tTrain error: %f" % train_error)
    print("\tTest error: %f" % test_error)
