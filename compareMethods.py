import numpy as np

# some distances for wanted comparison
METHODS = ["bhattacharyya", "euclidean", "manhattan", "chebysev", "momental_euclidean"]
method = -1


class Histogram:  # object that takes parameters and gives histogram and moments computed
    def __init__(self, model, max_n, distribution, logarithmic_scale=False, n_bins=100):
        # object with its parameters, histogram, bins, moments
        self.parameters = {"model": model, "max_n": max_n, "distribution": distribution,
                           "logarithmic": logarithmic_scale}
        with open(f'data/inf_{model}_{max_n}_{"_".join(distribution)}.txt', 'r') as f:  # read data
            array = [float(line) for line in f.readlines()]
        array = list(map(lambda x: x if logarithmic_scale else 10 ** x, array))  # transform if logarithmic scale
        max_value = 25 if logarithmic_scale else 6e7
        self.histogram, self.bins = np.histogram(array, n_bins, [0, max_value])  # make histogram from 0 to maxValue
        self.bins = np.array([i*max_value/n_bins for i in range(n_bins)])
        m = np.mean(self.histogram)
        n = 10  # number of computed central moments
        self.moments = np.array([m] + [np.sum((self.histogram - m) ** i) ** (1 / i) / i ** 5 for i in range(2, n + 1)])

    def __sub__(self, other):  # allow difference among Histogram objects using selected distance
        # if method == 0: return self.correlation(other)
        # elif method == 1: return self.intersection(other)
        if method == 0:
            return self.bhattacharyya(other)
        elif method == 1:
            return self.euclidean(other)
        elif method == 2:
            return self.manhattan(other)
        elif method == 3:
            return self.chebysev(other)
        else:
            return self.momental_euclidean(other)

    def correlation(self, other):  # correlation between histograms
        h1, h2 = self.histogram, other.histogram
        m1 = np.mean(h1)
        m2 = np.mean(h2)
        return - np.sum((h1 - m1) * (h2 - m2)) / np.sqrt(np.sum((h1 - m1) ** 2) * np.sum((h2 - m2) ** 2))

    def intersection(self, other):   # minimum of histograms
        h1, h2 = self.histogram, other.histogram
        return - np.sum(np.minimum(h1, h2))

    def bhattacharyya(self, other):  # bhattacharyya histogram distance
        h1, h2 = self.histogram, other.histogram
        m1 = np.mean(h1)
        m2 = np.mean(h2)
        return np.sqrt(max(0, 1 - np.sum(np.sqrt(h1 * h2)) / np.sqrt(m1 * m2 * len(h1) * len(h2))))

    def euclidean(self, other):   # euclidean distance between two histograms
        h1, h2 = self.histogram, other.histogram
        return np.sqrt(np.sum((h1 - h2) ** 2))

    def manhattan(self, other):   # manhattan distance between two histograms
        h1, h2 = self.histogram, other.histogram
        return np.sum(np.abs(h1 - h2))

    def chebysev(self, other):    # chebysev distance between two histograms
        h1, h2 = self.histogram, other.histogram
        return np.max(np.abs(h1 - h2))

    def momental_euclidean(self, other):   # euclidean distance between moments of two histograms
        return np.sqrt(np.sum((self.moments - other.moments) ** 2))
