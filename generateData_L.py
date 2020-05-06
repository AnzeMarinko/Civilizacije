from time import time  # to measure runtime
from numpy import array, histogram
from os import listdir  # to get list of files in directory
from models import get_point, distributions  # all defined models
from multiprocessing import Pool, freeze_support   # multi-threading

noIterations = 2e5  # number of generated points for each histogram
# all maximal numbers of civilisations
nrange = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
bin_no = 100  # number of bins in histograms

d = len(distributions)
# make set of all possible combinations for distributions (models 1 and 3 have 6 random parameters, 2 has 3)
parameters = set([((1, 3), (d1, d2, d3, d4, d5, d6)) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6 in range(d)] +
                 [((0, 2), (d1, d2, d3, 0, 0, 0)) for d1 in range(d) for d2 in range(d) for d3 in range(d)])
# make set of already generated combinations
open(f'collectedData/lin-parameters.txt', 'a').write("")
open(f'collectedData/parameters.txt', 'a').write("")
existing = {((1, 3), tuple([int(i) for i in par.split("_")[1:-1]])) if "2" not in par.split("_")[0] else
            ((0, 2), tuple([int(i) for i in par.split("_")[1:-1]]+[0, 0, 0])) for par in
            set(open("collectedData/lin-parameters.txt", "r").read().split("\n") +
                open("collectedData/parameters.txt", "r").read().split("\n"))}
# make set of not generated distributions
parameters.difference_update(existing)
print(len(parameters))
parameters = sorted(list(parameters))


def generate_by_n(par):  # generate histograms for all max. n-s at selected model and distributions
    if 2 in par[0]:  # model 2 has only 3 random variables
        (model, dist) = ((2, ), par[1][:3])
    else:
        (model, dist) = (par[0], par[1])
    name = [f"{m}_"+"_".join([str(p) for p in dist]) for m in model]  # make name for file
    t = time()
    hists = [[] for _ in model]  # make some initially empty lists
    linhists = [[] for _ in model]
    p = 0
    for maxN in nrange:
        # generate points distributed by model using selected distribution and maxN
        arr = array([get_point(maxN, dist, model) for _ in range(0, int(noIterations))])
        for m in range(len(model)):
            for logarithmic_scale in [True, False]:  # make it in logarithmic and linear scale
                # make histogram with selected number of bins, scale, from 0 to some big value
                hist, _ = histogram(arr[:, m] if logarithmic_scale else 10 ** arr, bin_no,
                                    [-1 if logarithmic_scale else 0, 12 if logarithmic_scale else 1e10])
                p = p+sum(hist)
                # make histogram from minValue to maxValue
                if logarithmic_scale:  # add generated data to lists
                    hists[m].append(hist)
                else:
                    linhists[m].append(hist)
    for m in range(len(model)):
        # write all histograms and moments and list of parameters in files to get processed data faster
        with open(f'data/lin-hists-{name[m]}.txt', 'w') as f:
            f.write(",".join([str(n) for n in nrange]) + "\n" +
                    "\n".join([",".join([str(x) for x in line]) for line in linhists[m]]))
        with open(f'data/log-hists-{name[m]}.txt', 'w') as f:
            f.write(",".join([str(n) for n in nrange]) + "\n" +
                    "\n".join([",".join([str(x) for x in line]) for line in hists[m]]))
    t = time() - t
    print(f"\tModel {model} distributed {dist}"
          f" generated cca. {p/(2*len(model)*len(nrange)*noIterations)*100:.2f} % points:\t\t {t:.4f} s")
    return t


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
