# Takes data from MovieLens dataset, extracts year
# of movie and age group of user

import pandas as pd
# Movie years

movies = pd.read_csv("./data/1M/movies.dat", sep='::', engine='python')
movies.columns = ['movieId', 'title', 'genres']


years = movies['title'].str.extract('\\((.*)\\)')
movie_years = pd.concat([movies['movieId'], years], axis=1)
movie_years.columns = ['movieId', 'year']

movie_years.to_csv('./data/1M/movie_years.csv', index=False)

