import numpy as np
import pandas as pd

num_genres = 10
num_features = 10

genres = set(range(1, num_genres + 1))

genre_labels = map(lambda x: "genre%d" % x, sorted(genres))
feature_labels = map(lambda x: "feature%d" % x, range(1, num_features+1))

items = pd.DataFrame()
items["age"] = np.rint(np.random.exponential(14, 1000))
item_genres = pd.DataFrame(np.random.binomial(1, 0.1, (1000, num_genres)))
item_genres.columns = genre_labels
item_features = pd.DataFrame(np.absolute(np.random.normal(0, 1, (1000, num_features))))
item_features.columns = feature_labels
items = pd.concat((items, item_genres, item_features), axis=1)

users = pd.DataFrame()
users["age"] = np.random.poisson(35, 1000)
users["seen"] = (np.random.poisson(4, 1000) * 20) + 20
user_genres = pd.DataFrame(np.random.binomial(1, 0.3, (1000, num_genres)))
user_genres.columns = genre_labels
user_features = pd.DataFrame(np.absolute(np.random.normal(0, 1, (1000, num_features))))
user_features.columns = feature_labels
users = pd.concat((users, user_genres, user_features), axis=1)

ratings_matrix = np.zeros((1000, 1000))


for i in range(0, 1000):
    row = users.ix[i]

    if i % 10 == 0:
        print i

    for j in range(0, int(row["seen"])):
        howfarback = np.random.poisson(row["age"]**0.5)
        whittle = items[items["age"] <= howfarback]
        my_genres = row[genre_labels].as_matrix()
        whittle = whittle.sample(20)
        its_genres = whittle[genre_labels].as_matrix()
        highest = np.argmax(np.dot(its_genres, my_genres))
        selected = whittle.iloc[highest]
        rating = np.dot(selected[feature_labels], row[feature_labels])
        ratings_matrix[i, selected.name] = np.rint(rating)


np.savetxt("./output/factorisations/synthetic/super.txt", ratings_matrix)



















