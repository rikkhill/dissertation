# Adds a csv file to the data directory of a specified dataset
# with relevance tags using that data directory's movie IDs

import pandas as pd
import numpy as np
from collections import defaultdict
import sys

try:
    sys.argv[1]
except IndexError:
    dim = 10
else:
    dim = int(sys.argv[1])

try:
    sys.argv[2]
except IndexError:
    factor_dir = "./output/factorisations/FF/"
else:
    factor_dir = sys.argv[2]

base_dir = "./data/FF/"

# Get a list of product descriptions in the training
# set, ordered by ProductId
ratings = pd.read_csv("%ssuperuser_purchases.csv" % base_dir)
base_products = pd.read_csv("%sproductinfo.csv" % base_dir,
                            sep=",",
                            error_bad_lines=False,
                            warn_bad_lines=True)
base_products.rename(columns={'ProductID': 'ProductId'}, inplace=True)
rated_products = ratings['ProductId'].unique().tolist()
base_products = base_products[base_products['ProductId'].isin(rated_products)]
base_products = base_products[["ProductId", "desc"]]
base_products = base_products.sort_values("ProductId")

# Get a list of the most common 1000 tags, ordered lexically
tags = pd.read_csv("%staginfo.csv" % base_dir)
tags.rename(columns={'ProductID': 'ProductId'}, inplace=True)
tag_counts = tags[["Tag", "ProductId"]].groupby("Tag").count()
tag_counts = tag_counts.reset_index().sort_values(by="ProductId", ascending=False)
tags = tag_counts["Tag"].head(1000).sort_values()

# Load the relevance matrix and the product matrix
rel_matrix = np.loadtxt("%sdimrelK%d.csv" % (factor_dir, dim), delimiter=" ")
product_matrix = np.loadtxt("%sdimproductsK%d.csv" % (factor_dir, dim), delimiter=" ")

(K, N) = rel_matrix.shape
basis_tags = []
product_examples = []

# Get examples from movies and examples from tags for each basis
for i in range(0, K):
    row_array = np.asarray(rel_matrix[i, :])
    toptwenty_tags = row_array.argsort()[-40:][::-1]
    basis_tags.append(toptwenty_tags)

    col_array = np.asarray(product_matrix[:, i])
    topten_products = col_array.argsort()[-14:][::-1]
    product_examples.append(topten_products)

movie_titles = []
for m in product_examples:
    movie_titles.append([base_products['ProductId'].iloc[n] for n in m])

# Get a list of lists of tags for each basis

tag_counts = defaultdict(int)
tag_words = [[] for i in range(0, K)]
count = 1

for i in basis_tags:
    count += 1
    for j in i:
        word = tags.iloc[j]
        tag_words[count - 2].append(word)
        tag_counts[word] += 1

best_tag_words = []
for t in tag_words:
    scoredict = dict((k, tag_counts[k]) for k in t if k in tag_counts)
    # find 5 lowest values in dictionary
    best_words = []
    for j in range(0, 10):
        w = min(scoredict, key=scoredict.get)
        best_words.append(w)
        del scoredict[w]
    best_tag_words.append(best_words)

for i in range(0, K):
    print("\nBasis %d" % (i + 1))
    print(best_tag_words[i])
    print(movie_titles[i])
