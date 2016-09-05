import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


base_dir = "./data/FF/"

ff = pd.read_csv("./data/FF/purchases.csv")
ff.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

product_info = pd.read_csv("%sproductinfo.csv" % base_dir,
                           sep=",",
                           error_bad_lines=False,
                           warn_bad_lines=False)


product_info.rename(columns={'ProductID': 'ProductId'}, inplace=True)

product_brands = product_info[["ProductId", "brand"]]

brand_purchases = pd.merge(ff, product_brands, on="ProductId")

top_users = brand_purchases[["UserID", "brand"]].groupby("UserID").agg("count")
top_users.reset_index(inplace=True)
top_users.sort_values(by="brand", ascending=False, inplace="true")
top_users = top_users["UserID"].head(5000).tolist()

brand_purchases = brand_purchases[brand_purchases["UserID"].isin(top_users)]
brand_purchases["value"] = 1

matrix = brand_purchases.pivot_table(aggfunc=lambda x: len(x.unique()),
                                     values="value",
                                     index="brand",
                                     columns="UserID")
matrix.fillna(0, inplace=True)
matrix = matrix.as_matrix()

print ((matrix > 0).sum() + 0.0) / (matrix.shape[0] * matrix.shape[1])
print matrix.shape

np.savetxt("./artefacts/bybrand.csv", matrix)

