import os
import numpy as np
from Converter import Converter, BIPOLAR_MODE

SAMPLE_FOLDER = "./learn/"


def threshold_function(i):
    return -1 if i < 0 else 1


class Hopfield:

    def __init__(self, width, height, act_func, folder=SAMPLE_FOLDER):
        self._folder = folder
        self._width = width
        self._height = height
        self._W = np.zeros((self._width * self._height, self._width * self._height))
        self._result = np.zeros((self._width * self._height, 1))
        self._sampels = []
        self._f = np.vectorize(act_func)
        self._h_dist = width * height
        self._h_result = np.zeros((self._width * self._height, 1))

    def teach(self):
        converter = Converter()
        for path, dirs, files in os.walk(self._folder):
            for fname in files:
                Xi = converter.to_nparray(self._folder + fname, BIPOLAR_MODE, (0, 0, 0), 10, True)
                self._W += Xi * Xi.T
                self._sampels.append(Xi)
        np.fill_diagonal(self._W, 0)

    def detect(self, img, max_iter):
        converter = Converter()
        X_star = converter.to_nparray(img, BIPOLAR_MODE, (0, 0, 0), 10, True)
        Yi = X_star
        count = 0
        while True:
            Yprev = Yi
            Yi = self._f(self._W @ Yi)
            if np.array_equal(Yi, Yprev) or count > max_iter:
                break
            count += 1
        self._result = Yi
        result = Yi.reshape((self._width, self._height))
        converter.to_bwimage(result, "result.jpg", BIPOLAR_MODE)

    def hamming_distance(self, save=False):
        s = 0
        converter = Converter()
        for arr in self._sampels:
            s = np.count_nonzero(arr != self._result)
            if s < self._h_dist:
                self._h_dist = s
                self._h_result = arr
        self._h_result = self._h_result.reshape((self._width, self._height))
        if(save):
            converter.to_bwimage(self._h_result, "h_result.jpg", BIPOLAR_MODE)
        return self._h_dist

