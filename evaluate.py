from performance.utils import *
import numpy as np
import pandas as pd

K = 10

factor_dir = "./output/factorisations/pmf/"

partitioned = pd.read_csv("./data/1M/partitioned_10pc.csv")

# Create training set with zeroed-out values for test elements
train = partitioned.copy()
train["rating"][train["training"] == 0] = 0

train = train.pivot(index="movieId", columns="userId", values="rating").fillna(0)
train.fillna(0, inplace=True)
train = train.as_matrix()

# Create test set with zeroed-out values for training elements
test = partitioned.copy()
test["rating"][test["training"] == 1] = 0

test = test.pivot(index="movieId", columns="userId", values="rating").fillna(0)
test.fillna(0, inplace=True)
test = test.as_matrix()

movie_matrix = np.loadtxt("%sdimmoviesK%d.csv" % (factor_dir, K), delimiter=" ")
user_matrix = np.loadtxt("%sdimusersK%d.csv" % (factor_dir, K), delimiter=" ")
#factor_dir = "/home/rikk/Workspace/testy/mf/"
#movie_matrix = np.loadtxt("%sdimmoviesK10.csv" % factor_dir, delimiter=" ")
#user_matrix = np.loadtxt("%sdimusersK10.csv" % factor_dir, delimiter=" ")

approx = np.dot(movie_matrix, user_matrix)

print scaled_f_norm(approx, train, scaled=False)
print scaled_f_norm(approx, test, scaled=False)
