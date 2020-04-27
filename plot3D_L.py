import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits import mplot3d


def draw_histograms3D(logarithmic_scale=True, model=1, distribution=(0, 0, 0, 0, 0, 0), bin_no=100):
    # draw in logarithmic scale on x, y axis
    parameters = sorted([tuple(int(i) for i in r) for r in [result[4:-4].split("_")
                                                                 for result in os.listdir('data')]],
                        key=lambda x: (x[0], x[2:], x[1]))
    # select the ones for drawing
    parameters = [test for test in parameters if test[0] == model and tuple(test[2:]) == distribution]

    bins = np.linspace(-2, 25, bin_no + 1) if logarithmic_scale else np.linspace(0, parameters[1][1], bin_no + 1)
    horSec = np.log10(np.array(sorted([par[1] for par in parameters])))
    Z = []

    for par in parameters:   # read data and make histograms
        with open(f'data/inf_{par[0]}_{par[1]}_{"_".join([str(i) for i in par[2:]])}.txt', 'r') as f:
            array = [float(line[0:-1]) if logarithmic_scale else 10**float(line[0:-1]) for line in f.readlines()]
        Z.append(np.histogram(array, bins)[0])
        print(par)

    X, Y = np.meshgrid(bins[:-1], horSec)
    fig = plt.figure()   # draw distributions
    ax = fig.add_subplot(111, projection='3d')
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap=plt.get_cmap("jet"), shade=True, alpha=0.8)

    ax.set_xlabel("log(L)")
    ax.set_ylabel("log(N)")
    ax.set_zlabel("# of hits")
    plt.title(f"Model {model} with {distribution} distribution in"
              f"{'logarithmic' if logarithmic_scale else 'linear'} scale")
    plt.show()

    print("done")
