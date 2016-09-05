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

#items = np.absolute(np.random.multivariate_normal(np.zeros(K), cov, num_items))
#users = np.absolute(np.random.multivariate_normal(np.zeros(K), cov, num_users)).T


items = np.absolute(np.random.normal(0, 0.7, item_dims))
users = np.absolute(np.random.normal(0, 0.7, user_dims))

print items.shape
print users.shape

prod = np.dot(items, users)

ratings = np.rint(prod)
ratings[ratings > 5] = 5
ratings[ratings < 1] = 1

res = np.histogram(ratings, bins=[1, 2, 3, 4, 5, 6])

print res
plt.hist(ratings.flatten(), bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], width=0.8, align='mid')
plt.xlim((0.5, 5.5))
plt.title("Distribution of synthetic ratings")

plt.show()