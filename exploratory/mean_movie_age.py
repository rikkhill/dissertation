# Plot mean age of movie rater

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

movie_years = pd.read_csv("./data/1M/movie_years.csv", index_col='movieId')

user_ages = pd.read_csv("./data/1M/users.dat",
                        sep='::',
                        engine='python',
                        header=None)
user_ages.columns = ['userId', 'gender', 'age', 'occupation', 'zip']
user_ages.set_index('userId', inplace=True)

ratings = pd.read_csv("./data/1M/ratings.dat",
                      sep='::',
                      engine='python',
                      header=None)

ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']

ratings_age = ratings.userId.apply(lambda x: user_ages.ix[x, 'age'] + 5)
ratings_year = ratings.movieId.apply(lambda x: movie_years.ix[x, 'year'])

year_ages = pd.concat([ratings_year, ratings_age], axis=1, ignore_index=True)
year_ages.columns = ["year", "age"]

#year_ages = year_ages[year_ages["year"] > 1971]

year_means = year_ages.groupby(["year"])["age"].mean()

year_means.plot()
plt.title("Release year against mean user age, all releases")
plt.show()