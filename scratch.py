import os
import re
import pandas as pd
from performance.utils import *
import matplotlib.pyplot as plt

label = "bnmf"
label2 = "agbnmf"

files = os.listdir("./output/results/")
pattern = re.compile("^%sK(\d*)t" % label)
pattern2 = re.compile("^%sK(\d*)t" % label2)

# Get all the values of K we have results for
vals = [int(m.group(1)) for f in files for m in [pattern.search(f)] if m]
vals = list(set(vals))
vals.sort()

vals2 = [int(m.group(1)) for f in files for m in [pattern2.search(f)] if m]
vals2 = list(set(vals2))
vals2.sort()

train = {}
test = {}

train2 = {}
test2 = {}

ix = [val for val in vals if val + 2 in vals2]

for K in ix:
    train[K] = pull_result("%sK%dtrain" % (label, K))
    test[K] = pull_result("%sK%dtest" % (label, K))
    print("%d rank factorisation" % K)
    print("\tTraining error: %f" % train[K].mean()[0])
    print("\tTest error: %f" % test[K].mean()[0])

vals2 = map(lambda x: x + 2, ix)

for K in vals2:
    train2[K] = pull_result("%sK%dtrain" % (label2, K))
    test2[K] = pull_result("%sK%dtest" % (label2, K))
    print("%d rank factorisation" % K)
    print("\tTraining error: %f" % train2[K].mean()[0])
    print("\tTest error: %f" % test2[K].mean()[0])

#train2 = train
#test2 = test
print ix
print vals2

data = pd.read_csv("./data/1M/partitioned_10pc.csv")
train_norm, test_norm = scale_norms(data, binary=True)

train_error = map(lambda x: train[x].mean()[0] / train_norm, ix)
test_error = map(lambda x: test[x].mean()[0] / test_norm, ix)

print test_error

train2_error = map(lambda x: train2[x].mean()[0] / train_norm, vals2)
test2_error = map(lambda x: test2[x].mean()[0] / test_norm, vals2)

#train2_error = train_error
#test2_error = test_error

print test2_error

df = pd.DataFrame()
df["K"] = ix
df["train"] = train_error
df["test"] = test_error
df["train2"] = train2_error
df["test2"] = test2_error

df = df[df["K"] <= 100]

ix = [val for val in vals if val in vals2]

print df

df.plot(x='K', y=["test", "test2"])
plt.title("Error curve: WNMF with Side Information: Gender")
plt.ylabel("Scaled error")
# df.plot(x='K', y=["test"])
plt.legend(labels=["Test Error", "Augmented Test Error"])
plt.show()

