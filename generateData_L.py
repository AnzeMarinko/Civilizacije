from time import time  # to measure runtime
from numpy import array, histogram, isnan, percentile, mean
from scipy.stats import moment  # to compute central moments
from os import listdir  # to get list of files in directory
from models import get_point, distributions, models  # all defined models
from multiprocessing import Pool, freeze_support

d = len(distributions)
# make set of all possible combinations for distributions (models 1 and 3 have 6 random parameters, 2 has 3)
parameters = set([(1, d1, d2, d3, d4, d5, d6) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6 in range(d)] +
                 [(2, d1, d2, d3, 0, 0, 0) for d1 in range(d) for d2 in range(d) for d3 in range(d)] +
                 [(1, d1, d2, d3, d4, d5, d6) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6 in range(d)])
# make set of already generated combinations
existing = {tuple([int(i) for i in par.split("_")[:-1]] + [0] * 3)[:7] for par in
            set(open("collectedData/lin-parameters.txt", "r").read().split("\n") +
                open("collectedData/parameters.txt", "r").read().split("\n"))}
# set of parameters that were generated but invalid output
open(f'collectedData/invalid-parameters.txt', 'a').write("")
invalid = {tuple([int(i) for i in par.split("_")[:-1]] + [0] * 3)[:7] for par in
           open(f'collectedData/invalid-parameters.txt', 'r').read().split("\n")[1:]}
# make set of not generated distributions
parameters.difference_update(existing)
parameters.difference_update(invalid)

noIterations = 1e6  # number of generated points for each histogram
# all maximal numbers of civilisations
nrange = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
bin_no = 64  # number of bins in histograms
n = 10  # number of computed central moments


def generate_by_n(par):  # generate histograms for all max. n-s at selected model and distributions
    if par[0] == 2:  # model 2 has only 3 random variables
        par = par[:4]
    (model, dist) = (par[0], par[1:])
    name = "_".join([str(p) for p in par])  # make name for file
    t = time()
    hists = []  # make some initially empty lists
    moments = []
    pars = []
    linhists = []
    linmoments = []
    linpars = []
    for maxN in nrange:
        # generate points distributed by model using selected distribution and maxN
        arr = array([get_point(maxN, dist, model) for _ in range(0, int(noIterations))])
        if percentile(arr, 95) > 15:  # if model gives too big probability to too big numbers break
            print("Overflow: more than 5 % for more than 10^15 years")
            break
        if isnan(arr).any():
            continue
        for logarithmic_scale in [True, False]:  # make it in logarithmic and linear scale
            # make histogram with selected number of bins, scale, from 0 to some big value
            hist, _ = histogram(arr if logarithmic_scale else 10 ** arr, bin_no,
                                [-1 if logarithmic_scale else 0, 20 if logarithmic_scale else 5e10])
            # make histogram from minValue to maxValue
            m = mean(hist)
            mom = moment(hist, range(2, n + 1))  # compute central moments
            if isnan(mom).any():
                print("nans: invalid central moments")
                continue
            if len(hist) != bin_no or len(mom) != n - 1:
                print("lens: invalid histogram or central moments")
                continue
            if logarithmic_scale:  # add generated data to lists
                hists.append(hist)
                moments.append((m, mom))
                pars.append(str(maxN))
            else:
                linhists.append(hist)
                linmoments.append((m, mom))
                linpars.append(str(maxN))
    # write all histograms and moments and list of parameters in files to get processed data faster
    if len(linpars) > 0:
        with open(f'data/lin-hists-{name}.txt', 'w') as f:
            f.write("\n".join([",".join([str(x) for x in line]) for line in linhists]))
        with open(f'data/lin-moments-{name}.txt', 'w') as f:
            f.write("\n".join([f"{m}," + ",".join([str(x) for x in line]) for m, line in linmoments]))
        with open(f'data/lin-parameters-{name}.txt', 'w') as f:
            f.write(",".join(linpars))
    if len(pars) > 0:
        with open(f'data/log-hists-{name}.txt', 'w') as f:
            f.write("\n".join([",".join([str(x) for x in line]) for line in hists]))
        with open(f'data/log-moments-{name}.txt', 'w') as f:
            f.write("\n".join([f"{m}," + ",".join([str(x) for x in line]) for m, line in moments]))
        with open(f'data/log-parameters-{name}.txt', 'w') as f:
            f.write(",".join(pars))
    t = time() - t
    if len(pars) + len(linpars) > 0:
        print(f"Model {model} distributed {dist}: {t:.4f} s")
    else:
        with open(f'collectedData/invalid-parameters.txt', 'a') as f:
            f.write("\n" + name)
    return t


def collect():
    # collect all generated histograms and moments in only 4 files (2 linear and 2 logarithmic)
    params = sorted([name[:-4].split("-")[-1] for name in listdir("data") if "log-p" in name])
    linparams = sorted([name[:-4].split("-")[-1] for name in listdir("data") if "lin-p" in name])
    with open("collectedData/lin-hists.txt", "w") as f:
        f.write("\n".join([open("data/lin-hists-" + par + ".txt", "r").read() for par in linparams]))
    with open("collectedData/lin-moments.txt", "w") as f:
        f.write("\n".join([open("data/lin-moments-" + par + ".txt", "r").read() for par in linparams]))
    with open("collectedData/hists.txt", "w") as f:
        f.write("\n".join([open("data/log-hists-" + par + ".txt", "r").read() for par in params]))
    with open("collectedData/moments.txt", "w") as f:
        f.write("\n".join([open("data/log-moments-" + par + ".txt", "r").read() for par in params]))
    # add list of parameters in right order
    with open("collectedData/lin-parameters.txt", "w") as f:
        f.write("\n".join(["\n".join([par + "_" + mn for mn in open("data/lin-parameters-" + par + ".txt",
                                                                    "r").read().split(",")]) for par in linparams]))
    with open("collectedData/parameters.txt", "w") as f:
        f.write("\n".join(["\n".join([par + "_" + mn for mn in open("data/log-parameters-" + par + ".txt",
                                                                    "r").read().split(",")]) for par in params]))
    print("Collecting data is done.")


def generate():
    t0 = time()
    # maximal number of new histograms:
    n_histograms = len(parameters) * len(list(nrange))
    freeze_support()
    # run generator of histograms on set of free parameters
    # use 10 threads, compute cumulative runtime for all histograms
    tsum = sum(Pool(10).map(generate_by_n, parameters))
    # number of random variables: model 1: 6, model 2: 3, model 3: 6
    tend = time() - t0  # runtime used after multi-threading
    print(f"\n\tTime used: {tend // 3600}h {tend // 60 % 60}min {tend % 60}s")
    print(f"\t{n_histograms} in {tend / tsum * 100:.2f} % of time\n")
    print("Generating data is done.")


if __name__ == "__main__":
    generate()
    collect()
