from numpy import meshgrid
from mpl_toolkits import mplot3d
from models import *


def draw_histograms3D(model=1, distribution=0, supermodel=1):
    # draw in logarithmic scale on x, y axis
    distribution = [distribution] * 6
    # get list of parameters
    data = get_point(distribution=distribution)[model-1]
    xlogn = xLogN2 if supermodel == 2 else xLogN
    X, Y = meshgrid(xLogL, xlogn)
    plt.figure(figsize=(5, 4), dpi=300, tight_layout=True)  # draw distributions
    title = ["Supermodel", "Model I", "Model II", "Model III", "Model IV"][model]
    d = distribution[0]
    ax = plt.subplot(1, 1, 1, projection='3d')
    # Plot the surface
    ax.plot_surface(X, Y, weight_dist(data, d, supermodel == 1, model), cmap=plt.get_cmap("jet"),
                    shade=True, alpha=0.8, linewidth=0)
    # ax.view_init(60, 270)
    ax.set_xlabel("log(L)")
    ax.set_ylabel("log(N)")
    ax.set_zlabel("density")
    plt.title(f"{title} distributed by {distributions[d]}")
    plt.savefig(f"out/{title.replace(' ', '-')}_{distributions[d]}_3D.png")

    plt.figure(figsize=(5, 4), dpi=300, tight_layout=True)
    plt.imshow(weight_dist(data, d, supermodel == 1, model)[::-1, :], cmap=plt.get_cmap("jet"),
               extent=[xLogL.min(), xLogL.max(), xlogn.min(), xlogn.max()])
    plt.xlabel("log(L)")
    plt.ylabel("log(N)")
    plt.title(f"{title} distributed by {distributions[d]}")
    plt.savefig(f"out/{title.replace(' ', '-')}_{distributions[d]}_heatmap.png")
    plt.figure(figsize=(10, 4), dpi=300, tight_layout=True)
    plt.suptitle(f"{title} distributed by {distributions[d]}")
    plt.subplot(131)
    plt.plot(xLogL, np.sum(weight_dist(data, d, supermodel == 1, model), 0))
    plt.title("Density function by L")
    plt.xlabel("log(L)")
    plt.ylabel("Density")
    plt.subplot(132)
    plt.plot(xLogL, 1 - np.cumsum(np.sum(weight_dist(data, d, supermodel == 1, model), 0)))
    plt.title("Survival function by L")
    plt.xlabel("log(L)")
    plt.ylabel("Probability")
    plt.grid(alpha=0.4)
    plt.subplot(133)
    plt.plot(xlogn, np.sum(weight_dist(data, d, supermodel == 1, model), 1))
    plt.title("Density function by N")
    plt.xlabel("log(N)")
    plt.ylabel("Density")
    plt.savefig(f"out/{title.replace(' ', '-')}_{distributions[d]}_properties.png")


def draw_histograms_N3D(model=1, distribution=0, supermodel=1):
    # draw in logarithmic scale on x, y axis
    distribution = [distribution] * 6
    # get list of parameters
    data = get_point(distribution=distribution)[model-1]
    X, Y = meshgrid(xLogL, xLogN2 if supermodel == 2 else xLogN)
    plt.figure(figsize=(10, 4), dpi=300)  # draw distributions
    for d in range(len(distributions)):
        ax = plt.subplot(1, 3, d+1, projection='3d')
        # Plot the surface
        ax.plot_surface(X, Y, weight_dist(data, d, supermodel == 1, model), cmap=plt.get_cmap("jet"),
                        shade=True, alpha=0.8)
        # ax.view_init(60, 270)
        ax.set_xlabel("log(L)")
        ax.set_ylabel("log(N)")
        ax.set_zlabel("density")
        plt.title(f"N distributed by {distributions[d]}")
    plt.savefig(f"out/distribute-N_3D.png")
