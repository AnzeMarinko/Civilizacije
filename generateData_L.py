from time import time  # to measure runtime
from numpy import array, histogram
import numpy as np
from os import listdir  # to get list of files in directory
from models import get_point, distributions  # all defined models
from multiprocessing import Pool, freeze_support  # multi-threading

noIterations = 2e5  # number of generated points for each histogram
# all maximal numbers of civilisations
nrange = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
bin_no = 100  # number of bins in histograms


def collect():
    # collect all generated histograms and moments in only 4 files (2 linear and 2 logarithmic)
    params = sorted([name[:-4].split("-")[-1] for name in listdir("data") if "log" in name])
    linparams = sorted([name[:-4].split("-")[-1] for name in listdir("data") if "lin" in name])
    lindata = [(p[0].split(","), "\n".join(p[1:])) for p in
               [open("data/lin-hists-" + par + ".txt", "r").read().split("\n") for par in linparams]]
    with open("collectedData/lin-hists.txt", "w") as f:
        f.write("\n".join([p for n, p in lindata]))
    with open("collectedData/lin-parameters.txt", "w") as f:
        f.write("\n".join(["\n".join([par + "_" + n for par, ns in
                                      zip(linparams, [n for n, p in lindata]) for n in ns])]))
    logdata = [(p[0].split(","), "\n".join(p[1:])) for p in
               [open("data/log-hists-" + par + ".txt", "r").read().split("\n") for par in params]]
    with open("collectedData/hists.txt", "w") as f:
        f.write("\n".join([p for n, p in logdata]))
    with open("collectedData/parameters.txt", "w") as f:
        f.write("\n".join(["\n".join([par + "_" + n for par, ns in
                                      zip(params, [n for n, p in logdata]) for n in ns])]))
    print("Collecting data is done.")


collect()  # to be sure that all generated data is already collected before the run

d = len(distributions)
# make set of all possible combinations for distributions (models 1 and 3 have 6 random parameters, 2 has 3, 4 has 7)
parameters = set([((1, 3), (d1, d2, d3, d4, d5, d6, 0)) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6 in range(d)] +
                 [((0, 2), (d1, d2, d3, 0, 0, 0, 0)) for d1 in range(d) for d2 in range(d) for d3 in range(d)] +
                 [((0, 4), (d1, d2, d3, d4, d5, d6, d7)) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6 in range(d) for d7 in range(d)])
# make set of already generated combinations
open(f'collectedData/lin-parameters.txt', 'a').write("")
open(f'collectedData/parameters.txt', 'a').write("")
existing = {((1, 3), tuple([int(i) for i in par.split("_")[1:-1]] + [0])) if "2" not in par.split("_")[0] else
            ((0, 2), tuple([int(i) for i in par.split("_")[1:-1]] + [0, 0, 0, 0])) if "4" not in par.split("_")[0] else
            ((0, 4), tuple([int(i) for i in par.split("_")[1:-1]]))
            for par in
            set(open("collectedData/lin-parameters.txt", "r").read().split("\n") +
                open("collectedData/parameters.txt", "r").read().split("\n")) if "5" not in par.split("_")[0]}

# make set of not generated distributions
parameters.difference_update(existing)
print(len(parameters))
# parameters = sorted(list(parameters))


def generate_by_n(par):  # generate histograms for all maxN-s at selected model and distributions
    if 2 in par[0]:  # model 2 has only 3 random variables etc.
        (model, dist) = ((2,), par[1][:3])
    elif 4 not in par[0]:
        (model, dist) = (par[0], par[1][:6])
    else:
        (model, dist) = ((4,), par[1])
    name = [f"{m}_" + "_".join([str(p) for p in dist]) for m in model]  # make name for file
    t = time()
    hists = [[] for _ in model]  # make some initially empty lists
    linhists = [[] for _ in model]
    for maxN in nrange if model[0] != 4 else [3000]:
        # generate points distributed by model using selected distribution and maxN
        arr = array([get_point(maxN, dist, model) for _ in range(0, int(noIterations))])
        for m in range(len(model)):
            for logarithmic_scale in [True, False]:  # make it in logarithmic and linear scale
                # make histogram with selected number of bins, scale, from 0 to some big value
                hist, _ = histogram(arr[:, m] if logarithmic_scale else 10 ** arr, bin_no,
                                    [-1 if logarithmic_scale else 0, 8 if logarithmic_scale else 1e8])
                # make histogram from minValue to maxValue
                if logarithmic_scale:  # add generated data to lists
                    hists[m].append(hist)
                else:
                    linhists[m].append(hist)
    for m in range(len(model)):
        # write all histograms and moments and list of parameters in files to get processed data faster
        with open(f'data/lin-hists-{name[m]}.txt', 'w') as f:
            f.write(",".join([str(n) for n in (nrange if model[0] != 4 else [3000])]) + "\n" +
                    "\n".join([",".join([str(x) for x in line]) for line in linhists[m]]))
        with open(f'data/log-hists-{name[m]}.txt', 'w') as f:
            f.write(",".join([str(n) for n in (nrange if model[0] != 4 else [3000])]) + "\n" +
                    "\n".join([",".join([str(x) for x in line]) for line in hists[m]]))
    t = time() - t
    print(f"\tModel {model} distributed {dist}"
          f" generated points:\t\t {t:.4f} s")
    return t


def generate():
    t0 = time()
    # maximal number of new histograms:
    n_histograms = len(parameters) * len(list(nrange))
    freeze_support()
    # run generator of histograms on set of free parameters
    # use 10 threads, compute cumulative runtime for all histograms
    tsum = sum(Pool(9).map(generate_by_n, parameters))
    # number of random variables: model 1: 6, model 2: 3, model 3: 6
    tend = time() - t0  # runtime used after multi-threading
    print(f"\n\tTime used: {tend // 3600}h {tend // 60 % 60}min {tend % 60}s")
    print(f"\t{n_histograms} in {tend / tsum * 100:.2f} % of time\n")
    print("Generating data is done.")


def interpolate(nInterpolations):  # only for logarithmic scale
    print("Interpolating ...")
    logarithmic_scale = True
    # all generated histograms:
    exist = np.array([[float(x) for x in par.split(",")] for par in
                      open(f'collectedData/{"" if logarithmic_scale else "lin-"}hists.txt', "r").read().split("\n")])
    new = np.array([])
    i = 0
    while i < nInterpolations:
        # take random two distributions out of already generated
        selected = exist[np.random.choice(range(len(exist)), size=2, replace=False)]
        if np.sum(np.abs(selected[0, :]-selected[1, :])) > 2e5:  # only if enough different
            p = 0.5+np.cbrt(np.random.random()/4-1/8)  # make some affine combination of those models
            # we want uniformly distributed so we took more probability for close to 0 or 1
            interp = np.dot(np.array([p, 1-p]), selected)
            new = np.append(new.reshape(-1, 100), interp.reshape(1, 100), axis=0)  # add new histogram to generated
            exist = np.append(exist.reshape(-1, 100), interp.reshape(1, 100), axis=0)
            i += 1
    with open(f'data/{"log-" if logarithmic_scale else "lin-"}hists-5_0.txt', 'w') as f:  # write new histograms in file
        f.write(",".join(["3000"] * nInterpolations) + "\n" +
                "\n".join([",".join([str(x) for x in line]) for line in new]))
    print("\t... is done!")


if __name__ == "__main__":
    # generate()
    for i in range(10):
        print(i+1)   # make few iterations where previous histograms from model 5 are lost to scatter more uniformly
        interpolate(2000)
        collect()
