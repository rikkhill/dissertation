from performance.utils import *

train = pull_result("wnmfK10train")
test = pull_result("wnmfK10test")

print(train.mean())
print(test.mean())
