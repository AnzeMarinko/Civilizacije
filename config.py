import numpy as np
import matplotlib.pyplot as plt
import os

if not os.path.exists("out"):
    os.mkdir("out")
plt.rcParams['figure.facecolor'] = 'white'

maxLogL = 7
logaritmic_steps_in_1logL = 16
logaritmic_steps_in_1logN = 16
noIterations = int(5e5)  # number of generated points for each N
noIterationsN = noIterations * 10   # number of generated points to distribute N
bin_no = maxLogL * logaritmic_steps_in_1logL  # number of bins in histograms
nrange = [10 ** (i / logaritmic_steps_in_1logN) for i in range(0, 3 * logaritmic_steps_in_1logN + 1)]  # N = 1:1e3
nrange2 = [10 ** (i / logaritmic_steps_in_1logN) for i in range(0, 8 * logaritmic_steps_in_1logN + 1)]  # N = 1:1e8
distributions = ["lognormal", "gauss", "loglinear"]  # all possible distributions
models = [1, 2, 3, 4]
xLogL = np.linspace(0, maxLogL, bin_no)
xLogN = np.log10(nrange)
xLogN2 = np.log10(nrange2)

data_folder = "data"
collected_folder = "collectedData"
if not os.path.exists(collected_folder):
    os.mkdir(collected_folder)


def sample_value(bound, distribution="lognormal", size=noIterations, i=0):
    mu, fromv, tov = bound
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
    return np.where((d > d * 0 + tov) + (d < d * 0 + fromv), sample_value((mu, fromv, tov), distribution, size, i + 1), d)


bounds = {"N": (0.5, np.log10(np.min(nrange)), np.log10(np.max(nrange))),
          "N2": (6, np.log10(np.min(nrange2)), np.log10(np.max(nrange2))),
          "R_*": (np.log10(2), 0, np.log10(5)),
          "f_p": (np.log10(0.9), -2, 0),
          "n_e": (np.log10(2), 0, np.log10(5)),
          "f_i": (np.log10(0.9), np.log10(0.2), 0),
          "f_c": (-1, -2, 0),
          "f_l": (np.log10(0.9), np.log10(0.2), 0),
          "f_a": (np.log10(3.6), -2, np.log10(25)),
          "f_b": (np.log10(0.081), -4.5, 0),
          "N_* n_e": (np.log10(5e11), np.log10(5e10), np.log10(5e12)),
          "f_g": (-1, np.log10(0.05), np.log10(0.15)),
          "f_{pm}": (-1, -2, np.log10(0.2)),
          "f_m": (-2, -2.5, -1.5),
          "f_j": (np.log10(0.8), -1, 0),
          "f_{me}": (-2, -2.5, -1.5)}

distNs = [np.histogram(sample_value(bounds["N"], distN, noIterationsN),
                       len(nrange), (np.log10(np.min(nrange)), np.log10(np.max(nrange))))[0] / noIterationsN
          for distN in distributions]
distNs2 = [np.histogram(sample_value(bounds["N2"], d, noIterationsN), len(nrange2),
                        (np.log10(np.min(nrange2)), np.log10(np.max(nrange2))))[0] / noIterationsN
           for d in ["loglinear", "uniform", "loguniform"]]


def weight_dist(distL, distN, supermodel_1=True, model=1):
    # Plot the surface
    dN = distNs2[distN] if not supermodel_1 and model == 3 else distNs[distN]
    lN = len(nrange2) if not supermodel_1 else len(nrange)
    return np.array([distL[i] * dN[i] if i < len(dN) else distL[0] * 0 for i in range(lN)])
