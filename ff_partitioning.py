import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ff = pd.read_csv("./data/FF/purchases.csv")
ff.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

prodcounts = ff[["UserID", "ProductId"]].groupby(["UserID"]).agg(["count"])

prodcounts.reset_index(inplace=True)
prodcounts.columns = ["UserID", "ProductCount"]

#superusers = prodcounts[(prodcounts["ProductCount"] >= 40) & (prodcounts["ProductCount"] < 100)]["UserID"].tolist()
superusers = prodcounts[(prodcounts["ProductCount"] >= 100)]["UserID"].tolist()

ff_superusers = ff[ff["UserID"].isin(superusers)]

tags = pd.read_csv("./data/FF/taginfo.csv")
tags.columns = ["ProductId", "Tag", "Type"]

tag_products = tags["ProductId"].unique().tolist()

ff_superusers = ff_superusers[ff_superusers["ProductId"].isin(tag_products)]

ff_superusers.to_csv("./data/FF/superuser_purchases.csv", index=False)


#training and test sets
holdout_pc = 10
ff_usercounts = ff_superusers.groupby("UserID")

test_set_idx = []

for user, group in ff_usercounts:
    holdout = int(len(group) * (holdout_pc / 100.0))
    test_set_idx += group["UnixTS"].nlargest(holdout).index.tolist()

test_set_idx = ff_superusers.index.isin(test_set_idx)

ff_superusers["training"] = np.array(~test_set_idx).astype(int)

ff_superusers.to_csv("./data/FF/superuser_10pc_partition_purchases.csv",
                     index=False)
