# Take hadoop MapReduce factorisations and produce relevance matrices for them

import pandas as pd
import numpy as np

K = 30

factor_dir = "./output/factorisations/FF/"

# Get lookup between product/user IDs and factor matrix indices
ff = pd.read_csv("./data/FF/superuser_purchases.csv")
ff.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
#ff = ff.head(30000)
ff = ff[["ProductId", "UserID"]].drop_duplicates()

productmap = pd.DataFrame()
productmap["ProductId"] = ff["ProductId"].sort_values().unique().copy()
productmap["ProductIndex"] = range(1, len(productmap) + 1)

usermap = pd.DataFrame()
usermap["UserID"] = ff["UserID"].sort_values().unique().copy()
usermap["UserIndex"] = range(1, len(usermap) + 1)


# Get factor matrices
w_matrix = np.loadtxt("%sdimproductsK%d.csv" % (factor_dir, K), delimiter=" ")
h_matrix = np.loadtxt("%sdimusersK%d.csv" % (factor_dir, K), delimiter=" ")

assert K == w_matrix.shape[1], "Dimensions must match"

# Load product tags

ff_tags = pd.read_csv("./data/FF/taginfo.csv")
ff_tags.columns = ["ProductId", "Tag", "Type"]

# Remove extraneous entries
product_id_list = productmap["ProductId"].unique().tolist()

#print(len(product_id_list))

ff_tags = ff_tags[ff_tags["ProductId"].isin(product_id_list)]

tag_counts = ff_tags[["Tag", "ProductId"]].groupby("Tag").count()
tag_counts = tag_counts.reset_index().sort_values(by="ProductId", ascending=False)

top_1000_tags = tag_counts["Tag"].head(1000).tolist()

ff_tags = ff_tags[ff_tags["Tag"].isin(top_1000_tags)]

print len(ff_tags["ProductId"].unique())
print len(ff_tags["Tag"].unique())

# Code for tags
tagmap = pd.DataFrame()
tagmap["Tag"] = ff_tags["Tag"].sort_values().unique().copy()
tagmap["TagId"] = range(1, len(tagmap) + 1)

# Add product indices, tag IDs and dummy value
ff_tags = pd.merge(ff_tags, productmap, on="ProductId")
ff_tags = pd.merge(ff_tags, tagmap, on="Tag")
ff_tags["value"] = 1
ff_tags = ff_tags[["ProductIndex", "TagId", "value"]].drop_duplicates()

tag_matrix = ff_tags.pivot(index="ProductIndex", columns="TagId", values="value")
tag_matrix.fillna(0, inplace=True)
tag_matrix = tag_matrix.as_matrix()

basis_relevance = np.dot(w_matrix.T, tag_matrix)
np.savetxt("%sdimrelK%d.csv" % (factor_dir, K), basis_relevance)