import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import operator
import matplotlib.pyplot as plt

def geometric_mean(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))

factor_dir = "./output/factorisations/bnmf10/"
comparison_dir = "./output/factorisations/wnmf/"

K = 10

runs = []

for n in range(0, 20):
    movie_matrix = np.loadtxt("%sdimmoviesK%dI%d.csv" % (factor_dir, K, n),
                              delimiter=" ")
    movie_examples = []

    for i in range(0, K):
        col_array = np.asarray(movie_matrix[:, i])
        topten_movies = col_array.argsort()[-10:][::-1]
        movie_examples.append(topten_movies)

    runs.append(movie_examples)


collect = set()

# 294 movies in the set


comparison = np.loadtxt("%sdimmoviesK%d.csv" % (comparison_dir, K), delimiter=" ")

active_bases = []
for i in range(0, K):
    col_array = np.asarray(comparison[:, i])
    topten_movies = col_array.argsort()[-10:][::-1]
    active_bases.append(topten_movies)


base_dir = "./data/1M/"
base_movies = pd.read_csv(base_dir + "movies.dat", sep="::", header=None, engine='python')
base_movies.columns = ["movieId", "title", "genre"]
ratings = pd.read_csv(base_dir + "ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
rated_movies = ratings['movieId'].unique().tolist()
base_movies = base_movies[base_movies['movieId'].isin(rated_movies)]


results = []

for m in range(0, len(runs)):

    active_run = runs[m]

    remainder = runs[:m] + runs[m + 1:]

    deciles = [None] * 10

    for num in range(1, 11):
        counts = defaultdict(int)
        for b in range(0, len(active_bases)):
            for r in remainder:
                for i in r:
                    if len(set(active_bases[b]) & set(i)) >= num:
                        counts[b] += 1

        deciles[num - 1] = counts

    results.append(len(deciles[9]))

    for i in deciles:
        print i

    for i in deciles:
        print len(i) / (K * 1.0)

print results

fig, ax = plt.subplots()


deciles = [
1.0,
1.0,
1.0,
1.0,
0.9,
0.8,
0.7,
0.5,
0.5,
0.1
]


ind = map(lambda x: x / 10.0, range(0, 10))
rects1 = ax.bar(ind, deciles, 0.1, color='#ffaaaa', edgecolor="none")
#rects2 = ax.bar(ind, female, width, color='b')

ax.set_ylabel('Proportion of bases seen before')
ax.set_xlabel("Threshold")
ax.set_title('MovieLens 1M BNMF, k=10 \nProportion of familiar bases as threshold increases\n k-fold cross-validation procedure')
plt.xlim((0.0, 1.0))
plt.ylim((0.0, 1.0))

plt.show()

