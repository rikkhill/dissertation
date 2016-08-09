import pandas as pd
import numpy as np

sales = pd.read_csv("./data/FF/superuser_purchases.csv")
sales.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)

tags = pd.read_csv("./data/FF/taginfo.csv")
tags.columns = ["ProductId", "Tag", "Type"]

sp = sales["ProductId"].unique()
tp = tags["ProductId"].unique()

print(sp.size)
print(tp.size)

print np.intersect1d(sp, tp, assume_unique=True).size
