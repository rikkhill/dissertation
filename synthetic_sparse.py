import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from pymf import *
from performance.utils import *


def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s




K = 30
num_items = 1000
num_users = 1000

# Set ratings dimensions
item_dims = (num_items, K)
user_dims = (K, num_users)

items = np.absolute(np.random.normal(0.0, 0.5, item_dims))
users = np.absolute(np.random.normal(0.0, 0.5, user_dims))

prod = np.dot(items, users)

ratings = np.rint(prod)

cutoff = sigmoid(prod - prod.mean()) / 4.9

vecprod = np.vectorize(lambda x: np.random.binomial(1, x))
mask = vecprod(cutoff)


ratings[ratings > 5] = 5
ratings[ratings < 1] = 1

ratings *= mask


np.savetxt("./data/synthetic/synth_blanks%d_%d_%d" % (num_items, num_users, K), ratings)

#print np.histogram(ratings, bins=[1, 2, 3, 4, 5, 6])

sys.exit(0)

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
    print(scaled_f_norm(approximation, noisy_ratings))



