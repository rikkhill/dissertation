import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

ff = pd.read_csv("./data/FF/purchases.csv")
ff.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)


salecounts = ff[["ProductId", "UserID"]].groupby("ProductId").agg("count")
salecounts.reset_index(inplace=True)
salecounts.columns = ["ProductId", "count"]
salecounts.sort_values("count", ascending=False, inplace=True)
superprods = salecounts[salecounts["count"] > 30]["ProductId"].unique().tolist()

# Filter out all products not sold 20 times
ff_superproducts = ff[ff["ProductId"].isin(superprods)]

prodcounts = ff_superproducts[["UserID", "ProductId"]].groupby(["UserID"]).agg(["count"])

prodcounts.reset_index(inplace=True)
prodcounts.columns = ["UserID", "ProductCount"]

#superusers = prodcounts[(prodcounts["ProductCount"] >= 40) & (prodcounts["ProductCount"] < 100)]["UserID"].tolist()
superusers = prodcounts[(prodcounts["ProductCount"] >= 20)]["UserID"].tolist()

ff_superusers = ff_superproducts[ff_superproducts["UserID"].isin(superusers)]

tags = pd.read_csv("./data/FF/taginfo.csv")
tags.columns = ["ProductId", "Tag", "Type"]

tag_products = tags["ProductId"].unique().tolist()

ff_superusers = ff_superusers[ff_superusers["ProductId"].isin(tag_products)]

#ff_superusers.to_csv("./data/FF/superuser_purchases.csv", index=False)


# Get top 7000 products from superusers

#training and test sets
holdout_pc = 10
ff_usercounts = ff_superusers.groupby("UserID")

test_set_idx = []

for user, group in ff_usercounts:
    holdout = int(len(group) * (holdout_pc / 100.0))
    test_set_idx += group["UnixTS"].nlargest(holdout).index.tolist()

test_set_idx = ff_superusers.index.isin(test_set_idx)

ff_superusers["training"] = np.array(~test_set_idx).astype(int)

unique_ix = ff_superusers[["ProductId", "UserID"]].drop_duplicates().index
ff_superusers = ff_superusers.ix[unique_ix]

ff_superusers.to_csv("./data/FF/dense_10pc_partition_purchases.csv",
                     index=False)
