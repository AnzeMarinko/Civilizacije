import matplotlib.pyplot as plt
import numpy as np


# scale and selected by histograms or moments
def draw_histograms(logarithmic_scale=True, by_histograms=False, bin_no=100):
    # plot only some selected distributions from each cluster
    if logarithmic_scale and not by_histograms:   # selected in clustered by moments in logarithmic scale
        # model, maxN, distribution, cluster, clustered by
        parameters = [(1, 3 ** 11, "uniform", 1, "moments"),
                      (3, 3 ** 11, "lognormal", 1, "moments"),
                      (3, 1, "lognormal", 1, "moments"),
                      (2, 1, "lognormal", 1, "moments"),
                      (2, 1, "loguniform", 1, "moments"),
                      (2, 3 ** 11, "loguniform", 1, "moments"),
                      (1, 1, "fixed", 2, "moments"),
                      (1, 1, "uniform", 2, "moments"),
                      (2, 1, "fixed", 3, "moments"),
                      (2, 3 ** 11, "uniform", 3, "moments"),
                      (1, 3 ** 11, "fixed", 4, "moments")]
    elif logarithmic_scale:   # selected in clustered by histograms in logarithmic scale
        # model, maxN, distribution, cluster, clustered by
        parameters = [(2, 3 ** 5, "fixed", 1, "histogrami"),
                      (3, 3 ** 11, "loguniform", 1, "histogrami"),
                      (3, 3 ** 11, "halfgauss", 1, "histogrami"),
                      (2, 1, "fixed", 2, "histogrami"),
                      (2, 3 ** 3, "fixed", 2, "histogrami"),
                      (3, 3 ** 5, "loguniform", 2, "histogrami"),
                      (1, 1, "fixed", 3, "histogrami"),
                      (3, 3 ** 6, "uniform", 3, "histogrami"),
                      (1, 3 ** 9, "fixed", 3, "histogrami"),
                      (2, 3 ** 11, "fixed", 4, "histogrami"),
                      (2, 3 ** 11, "loguniform", 4, "histogrami"),
                      (2, 3 ** 11, "lognormal", 4, "histogrami")]
    else:   # selected in clustered in linear scale
        # model, maxN, distribution, cluster, clustered by
        parameters = [(2, 3 ** 8, "fixed", 1, "histogrami/momenti"),
                      (1, 3 ** 7, "lognormal", 1, "histogrami/momenti"),
                      (1, 3 ** 8, "lognormal", 1, "histogrami/momenti"),
                      (2, 3 ** 4, "loguniform", 1, "histogrami/momenti"),
                      (2, 3 ** 7, "fixed", 2, "histogrami/momenti"),
                      (3, 1, "uniform", 2, "histogrami/momenti"),
                      (3, 3 ** 11, "uniform", 2, "histogrami/momenti"),
                      (2, 1, "fixed", 3, "histogrami/momenti"),
                      (2, 3 ** 6, "fixed", 3, "histogrami/momenti"),
                      (2, 3 ** 11, "fixed", 4, "histogrami/momenti"),
                      (1, 3 ** 11, "lognormal", 4, "histogrami/momenti"),
                      (2, 3 ** 3, "lognormal", 4, "histogrami/momenti"),
                      (2, 3 ** 11, "lognormal", 4, "histogrami/momenti"),
                      (2, 3 ** 11, "loguniform", 4, "histogrami/momenti")]

    # colours for clusters
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'b', 'g', 'r', 'm']

    for par in parameters:
        with open(f'data/inf_{par[0]}_{par[1]}_{par[2]}.txt', 'r') as f:  # read data
            array = [float(line) for line in f.readlines()]
        array = list(map(lambda x: x if logarithmic_scale else 10 ** x, array))  # transform if logarithmic scale
        maxDrawn = 30 if logarithmic_scale else 5000 * par[1]
        y, bins = np.histogram(array, bin_no, (0, maxDrawn))
        y = y / max(y)    # draw relative probabilities
        plt.plot(bins[:-1], y, colors[par[3] - 1], label=f'N={par[1]}, cluster {par[3]}, model {par[0]}, {par[2]}')
        plt.xlabel(f"Expected civilization longevity in {'log(years)' if logarithmic_scale else 'years'}")
        plt.ylabel("Relative probability")
    plt.title(f'Logarithmic scale distributions of models clustered by {"histograms" if by_histograms else "moments"}'
              if logarithmic_scale else "Linear scale distributions of models clustered")
    plt.legend(loc=1)
    plt.show()
