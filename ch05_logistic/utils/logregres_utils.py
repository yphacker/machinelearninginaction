# coding=utf-8
# author=yphacker

import numpy as np

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))