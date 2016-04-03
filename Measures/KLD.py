__author__ = 'Kirill Rudakov'
import numpy as np

def kld(p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,p * np.log(p / q), 0))
