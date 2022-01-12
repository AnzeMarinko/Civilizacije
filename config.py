import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists("out"):
    os.mkdir("out")
plt.rcParams['figure.facecolor'] = 'white'

noIterations = int(2e5)  # number of generated points for each histogram
maxLogL = 7
bin_no = maxLogL * 8  # number of bins in histograms
nrange = [10 ** (i / 16) for i in range(0, 3 * 16 + 1)]  # N = 1 ... 1000
nrange2 = [10 ** (i / 16) for i in range(0, 8 * 16 + 1)]  # N = 1 ... 100 000 000
distributions = ["lognormal", "gauss", "loglinear"]  # all possible distributions
models = [1, 2, 3, 4]
xLogL = np.linspace(0, maxLogL, bin_no)
xLogN = np.log10(nrange)
xLogN2 = np.log10(nrange2)

data_folder = "data"
collected_folder = "collectedData"


def sample_value(mu, fromv, tov, distribution="lognormal", size=noIterations, i=0):
    # random value from "distribution" distribution from "10**fromv" to "10**tov"
    d = None
    if distribution == "lognormal":
        mean = mu  # (tov + fromv) / 2  # half of interval
        sigma = (
            (tov - mean) if mu < 0 else (mean - fromv))  # divided by 2 so that 2*sigma expands along whole interval
        d = np.random.normal(mean, sigma, size)  # lognormal
    elif distribution == "loglinear":
        ad = np.random.uniform(0, 1, size)
        m1 = mu - fromv
        m2 = tov - fromv
        m3 = tov - mu
        first = np.sqrt(ad * m1 * m2) + fromv
        second = tov - np.sqrt((1 - ad) * m3 * m2)
        d = np.where(ad > m1 / m2, second, first)
    elif distribution == "uniform":
        # d = np.log10(np.random.uniform(10 ** fromv, 10 ** tov, size))
        sigma = (tov - mu) / 2  # divided by 2 so that 2*sigma expands along whole interval
        d = np.random.normal(mu, sigma, size)  # lognormal
    elif distribution == "loguniform":
        d = np.random.uniform(fromv, tov, size)
    elif distribution == "gauss":
        mean = 10 ** mu  # half of interval
        sigma = (10 ** tov - mean) / 1.5 if mu < 0 else (mean - 10 ** fromv)
        d = np.log10(np.abs(np.random.normal(mean, sigma, size)))  # lognormal
    if i > 10:
        return d
    return np.where((d > d * 0 + tov) + (d < d * 0 + fromv), sample_value(mu, fromv, tov, distribution, size, i + 1), d)


distNs = [np.histogram(sample_value(0.5, np.log10(np.min(nrange)), np.log10(np.max(nrange)), distN, noIterations * 10),
                       len(nrange), (np.log10(np.min(nrange)), np.log10(np.max(nrange))))[0] / noIterations / 10
          for distN in distributions]
distNs2 = [
    np.histogram(sample_value(6, np.log10(np.min(nrange2)), np.log10(np.max(nrange2)), "loglinear", noIterations * 10),
                 len(nrange2), (np.log10(np.min(nrange2)), np.log10(np.max(nrange2))))[0] / noIterations / 10,
    np.histogram(sample_value(6, np.log10(np.min(nrange2)), np.log10(np.max(nrange2)), "uniform", noIterations * 10),
                 len(nrange2), (np.log10(np.min(nrange2)), np.log10(np.max(nrange2))))[0] / noIterations / 10,
    np.histogram(sample_value(6, np.log10(np.min(nrange2)), np.log10(np.max(nrange2)), "loguniform", noIterations * 10),
                 len(nrange2), (np.log10(np.min(nrange2)), np.log10(np.max(nrange2))))[0] / noIterations / 10]


def weight_dist(distL, distN, supermodel=1, model=1):
    # Plot the surface
    dN = distNs2[distN] if supermodel == 2 and model == 3 else distNs[distN]
    lN = len(nrange2) if supermodel == 2 else len(nrange)
    return np.array([distL[i] * dN[i] if i < len(dN) else distL[0] * 0 for i in range(lN)])
