# Utilities for assessing performance of factorisations

import numpy as np
import pandas as pd


def scaled_f_norm(m_trained, m_comparison, scaled=True):
    mask = (m_comparison > 0).astype(int)
    # Ignore zero-values in comparison
    diff = (m_trained - m_comparison) * mask
    f_norm = np.sqrt(np.sum(diff * diff))

    # scale by cardinality of non-zero elements
    return f_norm / (np.sqrt(np.sum(m_comparison * m_comparison)) if scaled else 1)


# Append result `val` to filename `name`
def write_result(fname, val):
    with open("./output/results/" + fname, "a") as f:
        f.write(str(val) + "\n")


# Pull results data as a pandas series
def pull_result(fname):
    return pd.read_csv("./output/results/" + fname, header=None)


def augment_vector(v):

    age_range = [
        #(0, 5),
        #(5, 10),
        #(10, 15),
        #(15, 20),
        #(20, 25),
        #(25, 30),
        #(30, 35),
        #(35, 40),
        #(40, 45),
        #(45, 50),
        #(50, 55),
        #(55, 100) # Catch all older movies

        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50),
        (50, 100)
    ]

    stack = []

    for (a, b) in age_range:
        stack.append(((v >= a) & (v < b)).astype(int))

    return np.stack(stack, axis=0).T


# Return the training and test scaling norms for a dataframe
def scale_norms(df, binary=False):
    training = df.copy()

    training["rating"][training["training"] == 0] = 0
    training = training.pivot(index="movieId", columns="userId", values="rating")
    training.fillna(0, inplace=True)
    training = training.as_matrix()
    if binary:
        training[training > 0] = 1

    # Zero out training set data in test set
    test = df.copy()

    test["rating"][test["training"] == 1] = 0
    test = test.pivot(index="movieId", columns="userId", values="rating")
    test.fillna(0, inplace=True)
    test = test.as_matrix()
    if binary:
        test[test > 0] = 1

    training_norm = np.sqrt(np.sum(training * training))
    test_norm = np.sqrt(np.sum(test * test))

    return training_norm, test_norm
