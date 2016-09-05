# Adds a csv file to the data directory of a specified dataset
# with relevance tags using that data directory's movie IDs

import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


try:
    sys.argv[1]
except IndexError:
    dim = 12
else:
    dim = int(sys.argv[1])

try:
    sys.argv[2]
except IndexError:
    factor_dir = "./output/factorisations/apgwnmf/"
else:
    factor_dir = sys.argv[2]

base_dir = "./data/1M/"
base_movies = pd.read_csv(base_dir + "movies.dat", sep="::", header=None, engine='python')
base_movies.columns = ["movieId", "title", "genre"]

rel_matrix = np.loadtxt("%sdimrelK%d.csv" % (factor_dir, dim), delimiter=" ")
movie_matrix = np.loadtxt("%sdimmoviesK%d.csv" % (factor_dir, dim), delimiter=" ")
user_matrix = np.loadtxt("%sdimusersK%d.csv" % (factor_dir, dim), delimiter=" ")

tags = pd.read_csv("./data/genome/tags.dat", sep="\t", header=None)
tags.columns = ["tagId", "tag", "popularity"]

tr = pd.read_csv("./data/genome/tag_relevance.dat", sep="\t", header=None)
tr.columns = ["movieId", "tagId", "relevance"]

# We now need to remove all the movies from base_movies which
# aren't in the ratings matrix, because otherwise the indices
# won't align and it'll tell us Apocalypse Now is a cute
# animated children's film...
ratings = pd.read_csv(base_dir + "ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
rated_movies = ratings['movieId'].unique().tolist()
base_movies = base_movies[base_movies['movieId'].isin(rated_movies)]

(K, N) = rel_matrix.shape
basis_tags = []
movie_examples = []

movie_matrix = movie_matrix.T
movie_matrix = movie_matrix[2:, :]

print (K, N)
print movie_matrix.shape
# Get examples from movies and examples from tags for each basis
for i in range(0, K):
    row_array = np.asarray(rel_matrix[i, :])
    toptwenty_tags = row_array.argsort()[-30:][::-1]
    basis_tags.append(toptwenty_tags)

    col_array = np.asarray(movie_matrix[:, i])
    topten_movies = col_array.argsort()[-10:][::-1]
    movie_examples.append(topten_movies)

print topten_movies

movie_titles = []
for m in movie_examples:
    movie_titles.append([base_movies['title'].iloc[n] for n in m])

# Get a list of lists of tags for each basis

tag_counts = defaultdict(int)
tag_words = [[] for i in range(0, K)]
count = 1
for i in basis_tags:
    count += 1
    for j in i:
        word = tags['tag'].iloc[j]
        responsibility = rel_matrix[(count - 2), j]
        tag_words[count - 2].append((word, responsibility))
        tag_counts[word] += 1


"""
best_tag_words = []
for t in tag_words:
    scoredict = dict((k, tag_counts[k]) for k in t[0] if k in tag_counts)
    # find 5 lowest values in dictionary
    best_words = []
    for j in range(0, 10):
        w = min(scoredict, key=scoredict.get)
        best_words.append(w)
        del scoredict[w]
    best_tag_words.append(best_words)
"""

# remove stopwords

stopwords = [
    "good",
    "good soundtrack",
    "good acting",
    "great",
    "great ending",
    "great acting",
    "great movie",
    "original",
    "original plot",
    "story",
    "mentor",
    "fun movie",
    "dialogue",
    "imdb top 250",
    "destiny",
    "good action",
    "catastrophe",
    "runaway",
    "chase",
    "interesting",
    "very interesting",
    "vengeance"
]

trimmed_tags = map(lambda x: [e for e in x if e[0] not in stopwords], tag_words)

for i in range(0, K):
    print("\nBasis %d" % (i + 1))
    print("Exemplar movies:")
    for j in movie_titles[i]:
        print("\t%s \\\\" % j)
    print("High-responsibility keywords:")
    for j in trimmed_tags[i][0:20]:
        print("\t%s - %.2f \\\\" % j)


sys.exit()

user_matrix = user_matrix.T
attributes = user_matrix[:, :18].T
print attributes.shape

movie_genres = pd.read_csv("./data/1M/movie_genres.csv")
movie_genres = movie_genres[movie_genres["movieId"].isin(base_movies)]
genre_list = movie_genres.columns[3:]

for g in range(0, 18):
    print genre_list[g]
    for i in range(0, K):
        print("\tBasis %d: %.4f" % (i + 1, attributes[g, i]))


male = attributes[0, :].tolist()
female = attributes[1, :].tolist()

index = 16
label = genre_list[index - 1]
data = attributes[index - 1, :20]

fig, ax = plt.subplots()

ind = range(1, 21)
rects1 = ax.bar(ind, data, width=0.8, color='b', align="center")
#rects2 = ax.bar(ind, female, width, color='b')

ax.set_ylabel('Coefficient')
ax.set_xlabel("Basis")
ax.set_title('Coefficient by basis - "%s"' % label)

ax.set_xlim((0.5, 20.5))
#ax.set_ylim([0, 0.01])
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.show()
plt.xticks(np.arange(1, 12, 1.0))

