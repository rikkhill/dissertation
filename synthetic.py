import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from pymf import *
from performance.utils import *


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

K = 10
num_items = 100
num_users = 100

# Set ratings dimensions
item_dims = (num_items, K)
user_dims = (K, num_users)

# Make a random covariance matrix
cov = np.absolute(np.random.normal(0, 1, (K, K)))
cov = np.dot(cov, cov)
cov += np.identity(K)

items = np.absolute(np.random.multivariate_normal(np.zeros(K), cov, num_items))
users = np.absolute(np.random.multivariate_normal(np.zeros(K), cov, num_users)).T

print items.shape
print users.shape

prod = np.dot(items, users)

ratings = np.rint(prod)
ratings[ratings > 5] = 5
ratings[ratings < 1] = 1

# Create dropout process


np.savetxt("./data/synthetic/synth_mvnorm%d_%d_%d" % (num_items, num_users, K), ratings)

print (ratings>0).sum() / (ratings.shape[0] * ratings.shape[1] * 1.0)

#print np.histogram(ratings, bins=[1, 2, 3, 4, 5, 6])
"""
for i in map(lambda x: x / 10.0, range(1, 11)):
    print("Holdout: %.2f" % i)

    noisy_ratings = ratings.copy()
    noisy_ratings *= np.random.binomial(1, 1 - i, ratings.shape)

    # Show stuff
    #print noisy_ratings

    #(counts, bins) = np.histogram(noisy_ratings, bins=[0, 1, 2, 3, 4, 5, 6])
    #print (sum(counts[1:]) * 1.0) / sum(counts)

    #(n, bins, patches) = plt.hist(noisy_ratings.flatten(), bins)
    #plt.show()

    weight_matrix = np.ones(noisy_ratings.shape)

    nmf_model = WNMF(noisy_ratings, weight_matrix, num_bases=K, mask_zeros=True)
    nmf_model.factorize(niter=100, show_progress=False)

    items_matrix = nmf_model.W
    users_matrix = nmf_model.H

    approximation = np.dot(items_matrix, users_matrix)
    print(scaled_f_norm(approximation, ratings))
    print(scaled_f_norm(approximation, noisy_ratings))"""



