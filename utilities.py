import numpy as np
import ndd

def mutual_inf_nsb(x,y,ks):
    """
    Calculate mutual information using NSB method
    """
    ar = np.column_stack((x,y))
    mi = ndd.mutual_information(ar,ks)
    return np.log2(np.e)*mi #ndd returns nats - multiply by log2(e) to convert to bits