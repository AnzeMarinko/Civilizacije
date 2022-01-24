import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.facecolor'] = 'white'

# parameters of the program
maxLogL = 7  # maximal log(L) for histograms
logaritmic_steps_in_1logL = 16  # resolution for histograms in L - number of steps per one logarithmic unit
logaritmic_steps_in_1logN = 16  # resolution for histograms in N - number of steps per one logarithmic unit
noIterations = int(5e5)  # number of generated points for each N
noIterationsN = noIterations * 10   # number of generated points to distribute N
bin_no = maxLogL * logaritmic_steps_in_1logL  # number of bins in L for histograms
xLogL = np.linspace(0, maxLogL, bin_no)  # list of Ls
# list of Ns, one for each supermodel
nrange = [10 ** (i / logaritmic_steps_in_1logN) for i in range(0, 3 * logaritmic_steps_in_1logN + 1)]  # N = 1:1e3
nrange2 = [10 ** (i / logaritmic_steps_in_1logN) for i in range(0, 8 * logaritmic_steps_in_1logN + 1)]  # N = 1:1e8
xLogN = np.log10(nrange)
xLogN2 = np.log10(nrange2)
cmap = plt.get_cmap("viridis")

distributions = ["lognormal", "gauss", "loglinear"]  # all possible distributions
models = [1, 2, 3, 4]    # all possible models
data_folder = "data"  # folder for generated histograms
collected_folder = "collectedData"  # folder for collected data prepared for analysis
for folder in [data_folder, collected_folder, "out"]:  # make all folders for data and for output images
    if not os.path.exists(folder):
        os.mkdir(folder)


# return "size" of sampled points with "distribution" distribution in bounds "bound"
def sample_value(bound, distribution="lognormal", size=noIterations, i=0):
    mu, fromv, tov = bound   # peak, minimum and maximum
    d = None
    if distribution == "lognormal":
        sigma = (tov - mu) if mu < 0 else (mu - fromv)
        d = np.random.normal(mu, sigma, size)  # lognormal
    elif distribution == "loglinear":  # roof with linear distribution to and from peak
        ad = np.random.uniform(0, 1, size)
        m1 = mu - fromv
        m2 = tov - fromv
        m3 = tov - mu
        first = np.sqrt(ad * m1 * m2) + fromv
        second = tov - np.sqrt((1 - ad) * m3 * m2)
        d = np.where(ad > m1 / m2, second, first)
    elif distribution == "lognormal2":
        sigma = (tov - mu) / 2  # divided by 2 so that 2*sigma expands along whole interval
        d = np.random.normal(mu, sigma, size)  # lognormal
    elif distribution == "loguniform":
        d = np.random.uniform(fromv, tov, size)
    elif distribution == "gauss":
        sigma = (10 ** tov - 10 ** mu) / 1.5 if mu < 0 else (10 ** mu - 10 ** fromv)
        d = np.log10(np.abs(np.random.normal(10 ** mu, sigma, size)))  # lognormal
    out = (d > tov) + (d < fromv)
    if i < 10 and np.sum(out) > 0:  # try 10-times to get values in interval
        d[out] = sample_value(bound, distribution, np.sum(out), i + 1)
    if np.sum(d > tov) > 0:
        d[d > tov] = tov
    if np.sum(d < fromv) > 0:
        d[d < fromv] = fromv
    return d


# all parameters peaks, minimums and maximums in logarithmic scale
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

# histograms for different distributions of N
distNs = [np.histogram(sample_value(bounds["N"], distN, noIterationsN),
                       len(nrange), (np.log10(np.min(nrange)), np.log10(np.max(nrange))))[0] / noIterationsN
          for distN in distributions]
# histograms for Supermodel 2 for Model III
distNs2 = [np.histogram(sample_value(bounds["N2"], d, noIterationsN), len(nrange2),
                        (np.log10(np.min(nrange2)), np.log10(np.max(nrange2))))[0] / noIterationsN
           for d in ["loglinear", "lognormal2", "loguniform"]]


# return weighted distribution for selected distribution of N
def weight_dist(distL, distN, supermodel_1=True, model=1):
    dN = distNs2[distN] if not supermodel_1 and model == 3 else distNs[distN]
    lN = len(nrange2) if not supermodel_1 else len(nrange)
    return np.array([distL[i] * dN[i] if i < len(dN) else distL[0] * 0 for i in range(lN)])
