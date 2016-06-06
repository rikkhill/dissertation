# Utilities for assessing performance of factorisations

import numpy as np


def scaled_f_norm(m_trained, m_comparison):
    mask = (m_comparison > 0).astype(int)
    # Ignore zero-values in comparison
    m_trained[mask] = 0
    diff = m_trained - m_comparison
    f_norm = np.sqrt(np.sum(diff ** 2))

    # scale by cardinality of non-zero elements
    return f_norm / np.sum(mask)