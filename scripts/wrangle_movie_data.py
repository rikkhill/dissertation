# Takes data from MovieLens dataset, extracts year of movie
import pandas as pd

# Movie years
movies = pd.read_csv("./data/1M/movies.dat",
                     sep='::',
                     engine='python',
                     header=None)

movies.columns = ['movieId', 'title', 'genres']


years = movies['title'].str.extract('\\((\\d*)\\)')
movie_years = pd.concat([movies['movieId'], years], axis=1)
movie_years.columns = ['movieId', 'year']
movie_years.set_index('movieId', inplace=True)

movie_years.to_csv('./data/1M/movie_years.csv')

