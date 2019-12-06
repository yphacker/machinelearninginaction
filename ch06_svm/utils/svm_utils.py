# coding=utf-8
# author=yphacker

import numpy as np


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j
