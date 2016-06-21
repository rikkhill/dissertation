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

#ff = ff.head(10000)

ff["value"] = 1
ff = ff[["ProductId", "UserID", "value"]].drop_duplicates()

h5 = h5py.File('./output/h5py/fff.hdf5', 'w')

ff = ff.pivot(index="ProductId", columns="UserID", values="value")
ff.fillna(0, inplace=True)
h5['ff'] = ff.as_matrix()

# Factorise this

K = 10

(m, n) = h5['ff'].shape

#h5['W'] = np.random.random((m, K))
#h5['H'] = np.random.random((K, n))

model = BNMF(h5['ff'], num_bases=K)
#model.W = h5['W']
#model.H = h5['H']
model.factorize( niter=2, show_progress=True)

products = model.W
users = model.H
np.savetxt("./output/factorisations/FF/productsK%d.csv" % K, products)
np.savetxt("./output/factorisations/FF/usersK%d.csv" % K, users)
