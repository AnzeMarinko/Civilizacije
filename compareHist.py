import matplotlib.pyplot as plt
import numpy as np
import compareMethods
import os


def distance_matrices(logarithmic_scale=True, bin_no=100):
    # import list of parameters
    parameters = sorted([(int(r[1]), int(r[2]), r[-1]) for r in [result[:-4].split("_") for result in
                                                                 os.listdir('data')]],
                        key=lambda x: (x[0], x[2], x[1]))
    N = len(parameters)
    hists = []

    for par in parameters:   # make histograms
        hist = compareMethods.Histogram(par[0], par[1], par[2], logarithmic_scale, bin_no)
        hists.append(hist)

    fig = plt.figure()
    # use different methods
    for i in range(len(compareMethods.METHODS)):   # draw distance matrix for each method as image
        compareMethods.method = i
        distance_matrix = np.array([[h1-h2 for h1 in hists] for h2 in hists])

        ax = fig.add_subplot(2, round(len(compareMethods.METHODS)/2+0.5), i + 1)
        ax.set_title("%s" % compareMethods.METHODS[i])
        print(compareMethods.METHODS[i])
        plt.imshow(distance_matrix)
        plt.colorbar()
        plt.xlim([0, N])
        plt.ylim([0, N])
    plt.title(f"Distance matrices on histograms in {'logarithmic' if logarithmic_scale else 'linear'} scale")
    plt.show()
