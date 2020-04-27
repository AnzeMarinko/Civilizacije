from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.tri as mtri


# PCA transformation to draw the most informative graphs
def pca(data):
    m = np.mean(data, 0).reshape(1, -1)   # mean
    # correlation matrix
    C = sum([np.dot(line.reshape(-1, 1), line.reshape(1, -1)) for line in data]) / data.shape[1] - np.dot(m.T, m)
    eigenvalues, eigenvectors = np.linalg.eig(C)   # compute eigenvalue decomposition
    # first k-columns are more important, they make the most of variance
    return np.dot(eigenvectors.T, data.T - m.T).T, eigenvalues, eigenvectors, m.T  # transformed data, eigenpairs, mean


# print some properties of each cluster
def summarize(clusters, exact, k, models, maxNs):
    for i in range(k):   # for each cluster
        clust = [exact[p, :] for p in range(len(clusters)) if clusters[p] == i]
        print(f"\nCluster {i + 1}")
        print("Models:\n\t" + "\t\t ".join([str(model) for model in models]))   # distribution of models in cluster
        print("\t" + "\t\t ".join([str(len([c for c in clust if c[0] == model])) for model in models]))
        # distribution of maxN in cluster
        print("maxN:\n\t" + " \t".join([str(round(np.log10(maxN) * 100) / 100) for maxN in maxNs]))
        print("\t" + " \t\t".join([str(len([c for c in clust if c[1] == maxN])) for maxN in maxNs]))


# compute polynomial surface approximating cluster
def approximate(points, n):
    if n == 1:   # linear approximation
        X = np.concatenate([np.ones((1, points.shape[0]))[0, :], points[:, 0],
                            points[:, 1]]).reshape(-1, points.shape[0]).T
    elif n == 2:   # quadratic approximation
        X = np.concatenate([np.ones((1, points.shape[0]))[0, :], points[:, 0], points[:, 1],
                            points[:, 0] * points[:, 1], points[:, 0] ** 2,
                            points[:, 1] ** 2]).reshape(-1, points.shape[0]).T
    else:     # cubic approximation
        X = np.concatenate([np.ones((1, points.shape[0]))[0, :], points[:, 0], points[:, 1],
                            points[:, 0] * points[:, 1], points[:, 0] ** 2,
                            points[:, 1] ** 2, points[:, 0] ** 2 * points[:, 1], points[:, 0] * points[:, 1] ** 2,
                            points[:, 0] ** 3, points[:, 1] ** 3]).reshape(-1, points.shape[0]).T
    b = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, points[:, 2]))  # coefficients of surface
    text = f"\nx3 = {b[0]:.4e} + {b[1]:.4e} x1 + {b[2]:.4e} x2"     # surface coefficients as string
    if n > 1:
        text += f"\n\t   + {b[3]:.4e} x1 x2 + {b[4]:.4e} x1^2 + {b[5]:.4e} x2^2"
    if n > 2:
        text += f"\n\t   + {b[6]:.4e} x1^2 x2 + {b[7]:.4e} x1 x2^2 + {b[8]:.4e} x1^3 + {b[9]:.4e} x2^3"
    return b, text


# compute clusters, draw and print some properties
def cluster(logarithmic_scale=True, by_histograms=True, ks=None):
    if ks is None:
        ks = [4, 7, 10]
    name = "histogram" if by_histograms else "moment"
    # import data (parameters and than data)
    parameters = [(int(p[0]), int(p[-1]), p[1:-1]) for p in [par.split("_") for par in
                                                         open(f'collectedData/{"" if logarithmic_scale else "lin-"}'
                                                              f'parameters.txt', "r").read().split("\n")]]
    if by_histograms:
        hists = np.array([[float(x) for x in par.split(",")] for par in
                          open(f'collectedData/{"" if logarithmic_scale else "lin-"}hists.txt',
                          "r").read().split("\n")])
        data = hists
    else:
        moments = np.array([[float(x) for x in par.split(",")] for par in
                            open(f'collectedData/{"" if logarithmic_scale else "lin-"}moments.txt',
                                 "r").read().split("\n")])
        data = moments
    # make lists of parameters
    maxNs = sorted(list({a for _, a, _ in parameters}))
    models = sorted(list({a for a, _, _ in parameters}))
    # exact parameters before clustering
    exact = np.array([[a[0], a[1]] for a in parameters])

    # remove linear dependency between dimensions in data
    data, eigval, eigvec, meandata = pca(data)
    bins = np.array([i * (25 if logarithmic_scale else 6e7) / 64 for i in range(64)])

    print(f"\n\n\n{name}s:")
    plt.figure(figsize=(12, 12))
    # draw how important is each dimension in transformed data (eigenvalues) and
    # how is it generated from original data (eigenvectors)
    t = bins if name in "histogram" else [i + 1 for i in range(len(eigvec[:, 0]))]
    plt.subplot(221)
    plt.plot(eigval)
    plt.xlabel(f"Expected civilization longevity in {'log(years)' if logarithmic_scale else 'years'}")
    plt.ylabel("Relative probability")
    plt.title(f"Eigenvalues for transformed {name} space")
    plt.subplot(222)
    plt.plot(np.cumsum(eigval) / np.sum(eigval))
    plt.title(f"Explained variance by component for transformed {name} space")
    plt.subplot(223)
    plt.plot(np.log10(eigval + 1e-17))
    plt.title(f"Logarithm of eigenvalues for transformed {name} space")
    plt.subplot(224)
    plt.plot(t, np.abs(eigvec[:, 0]), label="1")
    plt.plot(t, np.abs(eigvec[:, 1]), label="2")
    plt.plot(t, np.abs(eigvec[:, 2]), label="3")
    plt.plot(t, np.abs(eigvec[:, 3]), label="4")
    plt.title(f"Weights for first dimensions in transformed {name} space")
    plt.legend(loc="best")
    plt.show()

    for k in ks:
        kmeans = KMeans(n_clusters=k).fit(data)
        clusters = kmeans.labels_
        means = kmeans.cluster_centers_
        means = (np.dot(eigvec, means.T) + np.dot(meandata, np.ones((k, 1)).T))  # stolpec je povprecje gruce
        if name in "histogram":
            means = (means.T / np.sum(means, 0).reshape(-1, 1)).T

        summarize(clusters, exact, k, models, maxNs)

        plt.figure(figsize=(5, 5))
        for i in range(k):
            plt.plot(t, means[:, i], label=f"{i+1}")
        plt.title(f"Mean {name}s for each cluster")
        plt.legend(loc="best")

        if name in "histograms":
            plt.figure(figsize=(5, 5))
            means = 1 - np.cumsum(means / np.sum(means, 0), 0)
            for i in range(k):
                plt.plot(t, means[:, i], label=f"{i + 1}")
            plt.title(f"Preživetvena funkcija za povprečje gruče")
            plt.legend(loc="best")

        plt.figure(figsize=(12, 12))
        mark = "P1^*os+x"
        colors = ["tab:"+c for c in ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]]
        plt.subplot(121, projection='3d')
        for model in models:
            trues = (exact[:, 0] == model)
            trues = trues * np.random.random(trues.shape[0]) > 0.995
            plt.plot(data[trues, 0],
                     data[trues, 1],
                     data[trues, 2],
                     linewidth=0, color=colors[model - 1])
            for maxN in maxNs:
                plt.plot(data[trues * (exact[:, 1] == maxN), 0],
                         data[trues * (exact[:, 1] == maxN), 1],
                         data[trues * (exact[:, 1] == maxN), 2],
                         marker=mark[4],
                         markersize=np.log10(maxN)+1,
                         linewidth=0, color=colors[model - 1])
        plt.title("Coloured by model - after Principal component analysis (PCA)")

        ax = plt.subplot(122, projection='3d')
        for i in range(k):
            points = data[clusters == i, :3]
            if name in "histograms":
                n = 3
                b, text = approximate(points, n)
                print(f"cluster {i + 1} surface:\n\t" + text)
                trues = np.random.random(points.shape[0]) > 0.99
                x = points[trues, 0]
                y = points[trues, 1]
                Z = 1 * b[0] + x * b[1] + y * b[2]
                if n > 1:
                    Z += x * y * b[3] + x ** 2 * b[4] + y ** 2 * b[5]
                if n > 2:
                    Z += x ** 2 * y * b[6] + x * y ** 2 * b[7] + x ** 3 * b[8] + y ** 3 * b[9]
                # Plot the surface.
                tri = mtri.Triangulation(x, y)
                ax.plot_trisurf(x, y, Z, triangles=tri.triangles, color=colors[i], linewidth=0, shade=True, alpha=0.8)
            trues = np.random.random(points.shape[0]) > 0.995
            ax.plot(points[trues, 0], points[trues, 1], points[trues, 2], label=i + 1, markersize=4, marker='.', linewidth=0)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
        plt.legend(loc="best")
        plt.title("Coloured by cluster")

        plt.show()
