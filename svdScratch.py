import os
import time
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
 
from svd_from_scratch import power_iteration_svd, jacobi_svd
 
 
IMAGE = "demo.jpeg"
RANKS = (5, 20, 100)

def load_grayscale(path):
    if os.path.exists(path):
        A = imread(path)
        return np.mean(A, -1).astype(np.float64)