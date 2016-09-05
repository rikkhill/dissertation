import pandas as pd
import numpy as np
from performance.utils import *
from pymf import *
import sys
import matplotlib.pyplot as plt


K = 10

data = pd.read_csv("./data/lastfm/userartist.csv")

print len(data)

data = data[["userId", "artistId"]].drop_duplicates()

print len(data)

data["value"] = 1

artistcounts = data[["userId", "artistId"]].groupby("artistId").agg("count")
artistcounts.reset_index(inplace=True)
artistcounts.columns = ["artistId", "count"]
artistcounts.sort_values(by="count", inplace=True, ascending=False)

print (artistcounts["count"] > 630).astype(int).sum()

top_artists = artistcounts[artistcounts["count"] > 630]["artistId"].tolist()

data = data[data["artistId"].isin(top_artists)]

print len(data)

usercounts = data[["userId", "artistId"]].groupby("userId").agg("count")
usercounts.reset_index(inplace=True)
usercounts.columns = ["userId", "count"]
usercounts.sort_values(by="count", inplace=True, ascending=False)

print (usercounts["count"] > 53).astype(int).sum()

top_users = usercounts[usercounts["count"] > 53]["userId"].tolist()

data = data[data["userId"].isin(top_users)]

print len(data)

data.to_csv("./data/lastfm/dense_user_artist.csv", index=False)

M = data.pivot(index="artistId", columns="userId", values="value")
M.fillna(0, inplace=True)
M = M.as_matrix()

print M.shape

print M.sum() / (M.shape[0] * M.shape[1] * 1.0)


nmf_model = BNMF(M, num_bases=K)
nmf_model.factorize(niter=60, show_progress=True)

items_matrix = nmf_model.W
users_matrix = nmf_model.H

approximation = np.dot(items_matrix, users_matrix)
print(scaled_f_norm(approximation, M))
np.savetxt("./output/factorisations/lastfm/dimmoviesK%d.csv" % K, items_matrix)
np.savetxt("./output/factorisations/lastfm/dimusersK%d.csv" % K, users_matrix)


