from performance.utils import *
import os

testfile = "test1.test"

try:
    os.remove(testfile)
except OSError:
    pass

for i in range(1, 11):
    write_result(testfile, i)

mean = pull_result(testfile).mean()[0]

assert mean == 6.0

os.remove("./output/results/%s" % testfile)