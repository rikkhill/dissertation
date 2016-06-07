# Utilities for assessing performance of factorisations

import numpy as np
import pandas as pd


def scaled_f_norm(m_trained, m_comparison, scaled=True):
    mask = (m_comparison > 0).astype(int)
    # Ignore zero-values in comparison
    diff = (m_trained - m_comparison) * mask
    f_norm = np.sqrt(np.sum(diff * diff))

    # scale by cardinality of non-zero elements
    return f_norm / (np.sum(mask) if scaled else 1)


# Append result `val` to filename `name`
def write_result(fname, val):
    with open("./output/results/" + fname, "a") as f:
        f.write(str(val) + "\n")


# Pull results data as a pandas series
def pull_result(fname):
    return pd.read_csv("./output/results/" + fname, header=None)
