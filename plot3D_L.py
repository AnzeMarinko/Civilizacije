import matplotlib.pyplot as plt
from numpy import meshgrid, linspace, log10, array
from mpl_toolkits import mplot3d


def draw_histograms3D(logarithmic_scale=True, model=3, distribution=(0, 0, 0, 0, 0, 0)):
    # draw in logarithmic scale on x, y axis
    # get list of parameters
    parameters = sorted([([int(i) for i in par.split("_")],
                          [float(h) for h in hist.split(",")])
                         for par, hist in
                         zip(open(f"collectedData/{'' if logarithmic_scale else 'lin-'}parameters.txt",
                                  "r").read().split("\n"),
                             open(f"collectedData/{'' if logarithmic_scale else 'lin-'}hists.txt",
                                  "r").read().split("\n"))])
    # filter parameters and histograms
    parameters = [(par[-1], hist) for par, hist in parameters if par[0] == model and tuple(par[1:-1]) == distribution]
    if len(parameters) > 0:  # draw histograms of selected model and distribution
        bins = linspace(-1 if logarithmic_scale else 0, 12 if logarithmic_scale else 1e10, len(parameters[0][1]))
        Z = array([hist for par, hist in parameters])
        n = [par for par, hist in parameters]
        X, Y = meshgrid(bins, log10(n))
        fig = plt.figure(figsize=(7, 7))  # draw distributions
        ax = fig.add_subplot(111, projection='3d')
        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap=plt.get_cmap("jet"), shade=True, alpha=0.8)
        ax.set_xlabel("log(L)")
        ax.set_ylabel("log(N)")
        ax.set_zlabel("# of hits")
        plt.title(f"Model {model} with {distribution} distribution in "
                  f"{'logarithmic' if logarithmic_scale else 'linear'} scale")
        plt.show()
        print("done")
