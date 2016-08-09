import pandas as pd
import numpy as np
import h5py
from pymf import *

ff = pd.read_csv("./data/FF/purchases.csv")
ff.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

# Pick 4000 users
# userids = ff.UserID.unique()
# users = np.random.choice(userids, 4000, replace=False).tolist()

# ff = ff[ff["UserID"].isin(users)]

ff = ff.head(30000)
# 1.29 million: 1290000
ff["value"] = 1
ff = ff[["ProductId", "UserID", "value"]].drop_duplicates()

h5 = h5py.File('./output/h5py/fff.hdf5', 'w')

#ff = ff.pivot(index="ProductId", columns="UserID", values="value")
#ff.fillna(0, inplace=True)


productmap = pd.DataFrame()
productmap["ProductId"] = ff["ProductId"].sort_values().unique().copy()
productmap["ProductIndex"] = range(1, len(productmap) + 1)

usermap = pd.DataFrame()
usermap["UserID"] = ff["UserID"].sort_values().unique().copy()
usermap["UserIndex"] = range(1, len(usermap) + 1)

ff = pd.merge(ff, productmap, on="ProductId")
ff = pd.merge(ff, usermap, on="UserID")

ff = ff[["ProductIndex", "UserIndex"]]

num_products = ff["ProductIndex"].unique().size
num_users = ff["UserIndex"].unique().size

arr = h5.create_dataset('ff', (num_products, num_users), chunks=True)

total = len(ff)

print total

for ix, pix, uix in ff.itertuples():
    if ix % 10000 == 0:
        print "%f%% - %d" % ((0.0 + ix) * 100 / total, ix)
    arr[pix-1, uix-1] = 1


# Factorise this

# Get indices
"""
trimmed_ff = ff[ff.index < limit]

trimmed_pix = sorted(trimmed_ff["ProductIndex"].unique().tolist())
trimmed_uix = sorted(trimmed_ff["UserIndex"].unique().tolist())

print(len(trimmed_pix), len(trimmed_uix))

arr2 = h5['ff'][trimmed_pix, trimmed_uix]
"""

K = 100


(m, n) = h5['ff'].shape

print (m, n)

#h5['W'] = np.random.random((m, K))
#h5['H'] = np.random.random((K, n))

model = BNMF(h5['ff'], num_bases=K)
#model.W = h5['W']
#model.H = h5['H']
model.factorize(niter=20, show_progress=True)

products = model.W
users = model.H
np.savetxt("./output/factorisations/FF/productsK%d.csv" % K, products)
np.savetxt("./output/factorisations/FF/usersK%d.csv" % K, users)
