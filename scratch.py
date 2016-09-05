import pandas as pd
import numpy as np
from performance.utils import *
from pymf import *
import sys
import matplotlib.pyplot as plt


K = 10

bx = pd.read_csv("./data/as/user_artist_data.txt", sep=" ")
bx.columns = ["userid", "artistid", "count"]
bx["count"] = bx["count"].apply(lambda x: x ** (1.0/3))

top_books = bx[["artistid", "count"]].groupby("artistid").agg("count")
top_books.reset_index(inplace=True)
top_books.columns = ["artistid", "count"]
top_books.sort_values("count", inplace=True, ascending=False)
top_books = top_books[top_books["count"] > 400]["artistid"].tolist()
print len(top_books)
bx = bx[bx["artistid"].isin(top_books)]
print bx["userid"].unique().size

print len(bx)

top_users = bx[["userid", "count"]].groupby("userid").agg("count")
top_users.reset_index(inplace=True)
top_users.columns = ["userid", "count"]
top_users.sort_values("count", inplace=True, ascending=False)
top_users = top_users[top_users["count"] > 400]["userid"].tolist()
print len(top_users)
bx = bx[bx["userid"].isin(top_users)]
print bx["artistid"].unique().size

print len(bx)

"""
#bx[bx["Book-Rating"] == 0] = 5 # 0 is implicit, so give it the mean rating
#bx = bx[bx["Book-Rating"] != 0]

usermap = pd.DataFrame(bx["User-ID"].unique())
usermap.columns = ["User-ID"]
usermap["userIndex"] = range(1, len(usermap) + 1)

bx = pd.merge(bx, usermap, on="User-ID")

bookmap = pd.DataFrame(bx["ISBN"].unique())
bookmap.columns = ["ISBN"]
bookmap["bookIndex"] = range(1, len(bookmap) + 1)

bx = pd.merge(bx, bookmap, on="ISBN")

bx = bx[["userIndex", "bookIndex", "Book-Rating"]]
bx = bx.drop_duplicates()

bx.columns = ["userIndex", "bookIndex", "rating"]

top_books = bx[["bookIndex", "rating"]].groupby("bookIndex").agg("count")
top_books.reset_index(inplace=True)
top_books.columns = ["bookIndex", "count"]
top_books.sort_values("count", inplace=True, ascending=False)
top_books = top_books[top_books["count"] > 10]["bookIndex"].tolist()
print len(top_books)
bx = bx[bx["bookIndex"].isin(top_books)]
print bx["userIndex"].unique().size

top_users = bx[["userIndex", "rating"]].groupby("userIndex").agg("count")
top_users.reset_index(inplace=True)
top_users.columns = ["userIndex", "count"]
top_users.sort_values("count", inplace=True, ascending=False)
top_users = top_users[top_users["count"] > 10]["userIndex"].tolist()
print len(top_users)
bx = bx[bx["userIndex"].isin(top_users)]
print bx["bookIndex"].unique().size

print len(bx)
"""

M = bx.pivot(index="artistid", columns="userid", values="count")
M.fillna(0, inplace=True)
M = M.as_matrix()

print M.shape

print M.sum() / (M.shape[0] * M.shape[1] * 1.0)


for i in range(0, 20):
    print "factorising  set %d" % i

    nmf_model = WNMF(M, np.ones(M.shape), num_bases=K, mask_zeros=True)
    nmf_model.factorize(niter=50, show_progress=True)
    movies = nmf_model.W
    users = nmf_model.H
    np.savetxt("./output/factorisations/as/dimmoviesK%dI%d.csv" % (K, i), movies)
    np.savetxt("./output/factorisations/as/dimusersK%dI%d.csv" % (K, i), users)




