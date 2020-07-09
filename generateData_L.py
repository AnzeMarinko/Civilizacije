from time import time  # to measure runtime
from numpy import array, histogram
import numpy as np
from os import listdir, mkdir, getcwd  # to get list of files in directory
from models import get_point, distributions  # all defined models
from multiprocessing import Pool, freeze_support  # multi-threading

noIterations = 2e5  # number of generated points for each histogram
bin_no = 100  # number of bins in histograms
folders = [10, 100, 1000, 10000]


def collect():
    for folder in folders:
        # collect all generated histograms and moments in only 4 files (2 linear and 2 logarithmic)
        params = sorted([name[:-4].split("-")[-1] for name in listdir(f"data/data{folder}") if "log" in name])
        linparams = sorted([name[:-4].split("-")[-1] for name in listdir(f"data/data{folder}") if "lin" in name])
        lindata = [(p[0].split(","), "\n".join(p[1:])) for p in
                   [open(f"data/data{folder}/lin-hists-" + par + ".txt", "r").read().split("\n") for par in linparams]]
        with open(f"collectedData/data{folder}/lin-hists.txt", "w") as g:
            g.write("\n".join([p for n, p in lindata]))
        with open(f"collectedData/data{folder}/lin-parameters.txt", "w") as g:
            g.write("\n".join(["\n".join([par + "_" + n for par, ns in
                                          zip(linparams, [n for n, p in lindata]) for n in ns])]))
        logdata = [(p[0].split(","), "\n".join(p[1:])) for p in
                   [open(f"data/data{folder}/log-hists-" + par + ".txt", "r").read().split("\n") for par in params]]
        with open(f"collectedData/data{folder}/hists.txt", "w") as g:
            g.write("\n".join([p for n, p in logdata]))
        with open(f"collectedData/data{folder}/parameters.txt", "w") as g:
            g.write("\n".join(["\n".join([par + "_" + n for par, ns in
                                          zip(params, [n for n, p in logdata]) for n in ns])]))
        print("Collecting data is done.")


def generate_by_n(par):  # generate histograms for all maxN-s at selected model and distributions
    if 2 in par[0]:  # model 2 has only 3 random variables etc.
        (model, dist, folder) = ((2,), par[1][:3], par[2])
    elif 4 not in par[0]:
        (model, dist, folder) = (par[0], par[1][:6], par[2])
    else:
        (model, dist, folder) = ((4,), par[1], par[2])
    nrange = [1, 2, 3, 4, 6, 8, 10] if folder == 10 else [1, 2, 4, 8, 16, 32, 64, 100] if folder == 100 else \
        [1, 3, 9, 27, 81, 243, 729, 1000] if folder == 1000 else [1, 4, 16, 64, 256, 1024, 4096, 10000]

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
        with open(f'data/data{folder}/lin-hists-{name[m]}.txt', 'w') as g:
            g.write(",".join([str(n) for n in (nrange if model[0] != 4 else [33333])]) + "\n" +
                    "\n".join([",".join([str(x) for x in line]) for line in linhists[m]]))
        with open(f'data/data{folder}/log-hists-{name[m]}.txt', 'w') as g:
            g.write(",".join([str(n) for n in (nrange if model[0] != 4 else [33333])]) + "\n" +
                    "\n".join([",".join([str(x) for x in line]) for line in hists[m]]))
    t = time() - t
    print(f"\tModel {model} distributed {dist}"
          f" generated points:\t\t {t:.4f} s")
    return t


def generate():
    mkdir(getcwd() + f'\\data')
    mkdir(getcwd() + f'\\collectedData')
    for folder in folders:
        mkdir(getcwd()+f'\\data\\data{folder}')
        mkdir(getcwd()+f'\\collectedData\\data{folder}')
        # all maximal numbers of civilisations
        nrange = [1, 2, 3, 4, 6, 8, 10] if folder == 10 else [1, 2, 4, 8, 16, 32, 64, 100] if folder == 100 else \
            [1, 3, 9, 27, 81, 243, 729, 1000] if folder == 1000 else [1, 4, 16, 64, 256, 1024, 4096, 10000]
        d = len(distributions)
        parameters = list(
            set(((1, 3), (d1, d2, d3, d4, d5, d6, 0), folder) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                for d4 in range(d) for d5 in range(d) for d6 in range(d)))[:d ** 3]
        parameters += [((0, 2), (d1, d2, d3, 0, 0, 0, 0), folder) for d1 in range(d) for d2 in range(d)
                       for d3 in range(d)]
        parameters += list(set(((0, 4), (d1, d2, d3, d4, d5, d6, d7), folder) for d1 in range(d) for d2 in range(d)
                               for d3 in range(d) for d4 in range(d) for d5 in range(d) for d6 in range(d)
                               for d7 in range(d)))[:d ** 3 * len(nrange)]
        parameters = list(set(parameters))
        print(len(parameters))
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
    for folder in folders:
        print("Interpolating ...")
        for logarithmic_scale in [True, False]:
            # all generated histograms:
            exist = np.array([[float(x) for x in par.split(",")] for par in
                              open(f'collectedData/data{folder}/{"" if logarithmic_scale else "lin-"}hists.txt',
                                   "r").read().split("\n")])
            new = np.array([])
            k = 0
            while k < nInterpolations:
                # take random two distributions out of already generated
                selected = exist[np.random.choice(range(len(exist)), size=2, replace=False)]
                if np.sum(np.abs(selected[0, :] - selected[1, :])) > noIterations:  # only if enough different
                    p = 0.5 + np.cbrt(
                        np.random.random() / 4 - 1 / 8) * 0.9  # make some affine combination of those models
                    # we want uniformly distributed so we took more probability for close to 0 or 1
                    interp = np.dot(np.array([p, 1 - p]), selected)
                    new = np.append(new.reshape(-1, 100), interp.reshape(1, 100),
                                    axis=0)  # add new histogram to generated
                    exist = np.append(exist.reshape(-1, 100), interp.reshape(1, 100), axis=0)
                    k += 1
            with open(f'data/data{folder}/{"log-" if logarithmic_scale else "lin-"}hists-5_0.txt', 'w') as g:
                # write new histograms in file
                g.write(",".join(["33333"] * nInterpolations) + "\n" +
                        "\n".join([",".join([str(x) for x in line]) for line in new]))
        print("\t... is done!")


if __name__ == "__main__":
    generate()
    collect()
    for i in range(10):
        print(i + 1)
        # make few iterations where previous histograms from model 5 are lost to scatter more uniformly
        interpolate(1000)
        collect()
