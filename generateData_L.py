import os
import time
import numpy as np
import compareMethods  # to compute histograms and moments
import models  # all defined models
import random

d = len(models.distributions)
distributions = ([(d1, d2, d3, d4, d5, d6) for d1 in range(d) for d2 in range(d) for d3 in range(d)
                  for d4 in range(d) for d5 in range(d) for d6
                  in range(d)], [(d1, d2, d3) for d1 in range(d) for d2 in range(d) for d3
                                 in range(d)])


def generate(noIterations=1e5, nrange=(2 ** i for i in range(15))):
    t = time.time()

    # noIterations = 1e6    # number of generated points for each selected parameters
    # nrange = [3**i for i in range(12)]   # list of possible values of maxN
    n_histograms = (2 * d ** 3 + 1) * d ** 3 * len(list(nrange))
    i = 0  # count generated distributions
    for model in models.models:
        dists = distributions[(model-1) % 2]
        for dist in dists:
            # number of random parameters: model 1: 6, model 2: 3
            n = 0
            m = len(list(nrange))
            for maxN in nrange:
                i += 1
                n += 1
                # generate points distributed by model using selected distribution and maxN
                filename = "data/inf_" + str(model) + "_" + str(maxN) + "_" + "_".join([str(i) for i in dist]) + ".txt"
                if filename in os.listdir('data'):
                    continue
                array = [models.get_point(maxN, dist, model) for _ in range(0, int(noIterations))]
                if np.percentile(array, 90) > 15:
                    i += m-n+1
                    print("Overflow")
                    break
                array = [str(i) for i in array if i < 18]
                with open(filename, 'w') as f:  # write generated points in file
                    f.write('\n'.join(array))
                print(f"{i / n_histograms ** 100:.2f}%: {i}/{n_histograms}\t\tFile: " + filename +
                      " created. no of points:" + str(len(array)))
    print('\tdone')
    t = time.time() - t  # print runtime
    print(f"\n\tTime used: {t // 3600}h {t // 60 % 60}min {t % 60}s\n\n")


def collect(bin_no=64):
    parameters = sorted([(int(r[1]), int(r[2]), r[3:]) for r in [result[:-4].split("_") for result
                                                                 in os.listdir('data')]],
                        key=lambda x: (x[0], x[2], x[1]))
    hists = []
    moments = []

    for logarithmic_scale in [True, False]:  # make it in logarithmic and linear scale
        for par in parameters:  # add histogram and moments to collecting lists
            hist = compareMethods.Histogram(par[0], par[1], par[2], logarithmic_scale, bin_no)
            hists.append(hist.histogram)
            moments.append(hist.moments)
        # write all histograms and moments and list of parameters in files to get processed data faster
        with open(f'collectedData/{"" if logarithmic_scale else "lin-"}hists.txt', 'w') as f:
            f.write("\n".join([",".join([str(x) for x in line]) for line in hists]))
        with open(f'collectedData/{"" if logarithmic_scale else "lin-"}moments.txt', 'w') as f:
            f.write("\n".join([",".join([str(x) for x in line]) for line in moments]))
        with open(f'collectedData/{"" if logarithmic_scale else "lin-"}parameters.txt', 'w') as f:
            f.write("\n".join([",".join([str(par[i]) for i in range(3)]) for par in parameters]))
