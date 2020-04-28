import matplotlib.pyplot as plt
import numpy as np


# scale and selected by histograms or moments
def draw_histograms(clusters, histograms, logarithmic_scale=True, by_histograms=False):
    # plot only some selected distributions from each cluster
    # colours for clusters
    colors = ["tab:"+c for c in ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]]
    bins = np.linspace(-1 if logarithmic_scale else 0, 20 if logarithmic_scale else 5e10, len(histograms[0]))
    for c, hist in zip(clusters, histograms):
        hist = hist / max(hist)    # draw relative probabilities and colour by cluster
        plt.plot(bins, hist, colors[c-1])
        plt.xlabel(f"Expected civilization longevity in {'log(years)' if logarithmic_scale else 'years'}")
        plt.ylabel("Relative probability")
    plt.title(f'Logarithmic scale distributions of models clustered by {"histograms" if by_histograms else "moments"}'
              if logarithmic_scale else "Linear scale distributions of models clustered")
    plt.show()
