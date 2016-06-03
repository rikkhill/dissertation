import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols


try:
    sys.argv[1]
except IndexError:
    K = 10
else:
    K = int(sys.argv[1])

try:
    sys.argv[2]
except IndexError:
    fact_dir = "./output/factorisations/bnmf/"
else:
    fact_dir = int(sys.argv[2])

base_dir = "./data/1M/"

# Load movie factor matrix
movie_matrix = np.loadtxt("%sdimmoviesK%d.csv" % (fact_dir, K), delimiter=" ")

# Get a list of all relevant movies
ratings = pd.read_csv(base_dir + "ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
rated_movies = ratings['movieId'].unique().tolist()

# Load movie years
movie_years = pd.read_csv(base_dir + "movie_years.csv")
# strip non-relevant entries
movie_years = movie_years[movie_years["movieId"].isin(rated_movies)]

movie_examples = []

# Sample 50 movies from each basis
for i in range(0, K):
    col_array = np.asarray(movie_matrix[:, i])

    print("Responsibility of basis %d: %d" % (i, np.linalg.norm(col_array)))

    top_n_movies = col_array.argsort()[-50:][::-1]
    movie_examples.append(top_n_movies)

sample_years = []
for i in movie_examples:
    sample_years.append(map(lambda x: 2000 - movie_years['year'].iloc[x], i))

factor_samples = pd.DataFrame.transpose(pd.DataFrame(sample_years))
factor_samples.columns = range(1, K + 1)

# There is almost certainly a better way to do this, but Pandas is horrendous
# and will do what it wants
factor_frame = pd.DataFrame.transpose(factor_samples).stack().reset_index()
factor_frame.columns = ['factor', 'index', 'year']

# Carry out ANOVA
fitted = ols('year ~ C(factor)', data=factor_frame).fit()
print(sm.stats.anova_lm(fitted, typ=2))
# print(fitted.summary())


# Make it look nice
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.xaxis.tick_top()
factor_samples.boxplot(vert=0,
                       return_type='axes',
                       column=list(reversed(range(1, K + 1))))
plt.show(fig)
