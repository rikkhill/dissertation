import pandas as pd
import numpy as np
import h5py
from pymf import *

ff = pd.read_csv("./data/FF/dense_10pc_partition_purchases.csv")


productmap = pd.DataFrame()
productmap["ProductId"] = ff["ProductId"].sort_values().unique().copy()
productmap["ProductIndex"] = range(1, len(productmap) + 1)

usermap = pd.DataFrame()
usermap["UserID"] = ff["UserID"].sort_values().unique().copy()
usermap["UserIndex"] = range(1, len(usermap) + 1)

ff = pd.merge(ff, productmap, on="ProductId")
ff = pd.merge(ff, usermap, on="UserID")
ff["value"] = 1
ff = ff[["ProductIndex", "UserIndex", "value"]].drop_duplicates()

# Create matrix
ff = ff.pivot(index="ProductIndex", columns="UserIndex", values="value")
ff.fillna(0, inplace=True)
matrix = ff.as_matrix()

print matrix.shape

# Factorise this

K = 100
model = BNMF(matrix, num_bases=K)
model.factorize(niter=55, show_progress=True)

products = model.W
users = model.H
np.savetxt("./output/factorisations/FF_dense/dimproductsK%d.csv" % K, products)
np.savetxt("./output/factorisations/FF_dense/dimusersK%d.csv" % K, users)
