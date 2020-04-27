import time
import numpy as np
import scipy.stats
import os
import models  # all defined models
import multiprocessing as mp

d = len(models.distributions)
distributions = ([(d1, d2, d3, d4, d5, d6) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6
                  in range(d)], [(d1, d2, d3) for d1 in range(d) for d2 in range(d) for d3
                                 in range(d)])

noIterations = 1e6
nrange = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
bin_no = 64
n = 10    # number of computed central moments
pow = 1/np.array(range(2, n+1))**4


def generateByN(par):
    (model, dist) = par
    name = "_".join([str(model)] + [str(s) for s in dist])
    if [1 for file in os.listdir("data") if name in file]:
        return 0
    t = time.time()
    hists = []
    moments = []
    pars = []
    linhists = []
    linmoments = []
    linpars = []
    for maxN in nrange:
        # generate points distributed by model using selected distribution and maxN
        array = np.array([models.get_point(maxN, dist, model) for _ in range(0, int(noIterations))])
        if np.percentile(array, 95) > 15:
            print("Overflow")
            break
        if np.isnan(array).any():
            continue
        array = array[array < 20]
        for logarithmic_scale in [True, False]:  # make it in logarithmic and linear scale
            histogram, _ = np.histogram(array if logarithmic_scale else 10 ** array,
                                        bin_no, [0, 20 if logarithmic_scale else 6e7])
            # make histogram from 0 to maxValue
            m = np.mean(histogram)
            moment = scipy.stats.moment(histogram, range(2, n+1))*pow
            if np.isnan(moment).any():
                print("nans")
                continue
            if len(histogram) != bin_no or len(moment) != n-1:
                print("lens")
                continue
            if logarithmic_scale:
                hists.append(histogram)
                moments.append((m, moment))
                pars.append(str(maxN))
            else:
                linhists.append(histogram)
                linmoments.append((m, moment))
                linpars.append(str(maxN))
    # write all histograms and moments and list of parameters in files to get processed data faster
    if len(linpars) > 0:
        with open(f'data/lin-hists-{name}.txt', 'w') as f:
            f.write("\n".join([",".join([str(x) for x in line]) for line in linhists]))
        with open(f'data/lin-moments-{name}.txt', 'w') as f:
            f.write("\n".join([f"{m},"+",".join([str(x) for x in line]) for m, line in linmoments]))
        with open(f'data/lin-parameters-{name}.txt', 'w') as f:
            f.write(",".join(linpars))
    if len(pars) > 0:
        with open(f'data/log-hists-{name}.txt', 'w') as f:
            f.write("\n".join([",".join([str(x) for x in line]) for line in hists]))
        with open(f'data/log-moments-{name}.txt', 'w') as f:
            f.write("\n".join([f"{m},"+",".join([str(x) for x in line]) for m, line in moments]))
        with open(f'data/log-parameters-{name}.txt', 'w') as f:
            f.write(",".join(pars))
    t = time.time()-t
    if len(pars) + len(linpars) > 0:
        print(f"Model {model} distributed {dist}: {t:.4f} s")
    return t


def collect():
    parameters = sorted([name[:-4].split("-")[-1] for name in os.listdir("data") if "log-p" in name])
    linparameters = sorted([name[:-4].split("-")[-1] for name in os.listdir("data") if "lin-p" in name])
    with open("collectedData/lin-hists.txt", "w") as f:
        f.write("\n".join([open("data/lin-hists-"+par+".txt", "r").read() for par in linparameters]))
    with open("collectedData/lin-moments.txt", "w") as f:
        f.write("\n".join([open("data/lin-moments-" + par + ".txt", "r").read() for par in linparameters]))
    with open("collectedData/hists.txt", "w") as f:
        f.write("\n".join([open("data/log-hists-" + par + ".txt", "r").read() for par in parameters]))
    with open("collectedData/moments.txt", "w") as f:
        f.write("\n".join([open("data/log-moments-" + par + ".txt", "r").read() for par in parameters]))
    # add list of parameters
    with open("collectedData/lin-parameters.txt", "w") as f:
        f.write("\n".join(["\n".join([par+"_"+mn for mn in open("data/lin-parameters-" + par + ".txt",
                                                                "r").read().split(",")]) for par in linparameters]))
    with open("collectedData/parameters.txt", "w") as f:
        f.write("\n".join(["\n".join([par + "_" + mn for mn in open("data/log-parameters-" + par + ".txt",
                                                                    "r").read().split(",")]) for par in parameters]))
    print("Generating and collecting is done.")


def generate():
    t0 = time.time()
    n_histograms = (2 * d ** 3 + 1) * d ** 3 * len(list(nrange))
    mp.freeze_support()
    with mp.Pool(5) as p:
        tsum = sum(p.map(generateByN, ((model, dist) for model in models.models
                                       for dist in distributions[(model-1) % 2])))
        # number of random variables: model 1: 6, model 2: 3, model 3: 6
    tend = time.time() - t0  # print runtime
    print(f"\n\tTime used: {tend // 3600}h {tend // 60 % 60}min {tend % 60}s\n\t{n_histograms} in {tend/tsum*100:.2f} "
          f"% of time\n")


if __name__ == "__main__":
    mp.freeze_support()
    generate()
    collect()
