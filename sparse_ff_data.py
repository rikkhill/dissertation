import pandas as pd

# Import data
# Write to sparse matrix file
ff = pd.read_csv("./data/FF/purchases.csv")
ff.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
# Make surrogate movie index for sparse matrix format
moviemap = pd.DataFrame()
moviemap["movieId"] = ff["movieId"].sort_values().unique().copy()
moviemap["movieIndex"] = range(1, len(moviemap) + 1)

training = pd.merge(ff, moviemap, on="movieId")

# Remove test set
training = training[training["training"] == 1]

# Remove irrelevant columns
training = training[["movieIndex", "userId", "rating"]]

# Write to CSV
training.to_csv("./data/1M/sparsefile_ff.txt",
                sep="\t",
                header=False,
                index=False)
