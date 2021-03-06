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

X = ratings_age.as_matrix()
X = sm.add_constant(X)

y = ratings_year.as_matrix()

res = sm.OLS(y, X).fit()
prstd, iv_l, iv_u = wls_prediction_std(res)

fig, ax = plt.subplots(figsize=(8,6))

ax.plot(X, y, 'o', label="Data")
ax.plot(X, res.fittedvalues, 'r--.', label="Predicted")
ax.plot(X, iv_u, 'r--')
ax.plot(X, iv_l, 'r--')
legend = ax.legend(loc="best")
plt.show(fig)