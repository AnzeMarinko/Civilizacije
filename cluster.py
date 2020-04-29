from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull


# PCA transformation to draw the most informative graphs
def pca(data):
    m = np.mean(data, 0).reshape(1, -1)  # mean
    # correlation matrix
    C = sum(np.dot(line.reshape(-1, 1), line.reshape(1, -1)) for line in data) / data.shape[1] - np.dot(m.T, m)
    eigenvalues, eigenvectors = np.linalg.eig(C)  # compute eigenvalue decomposition
    # first k-columns are more important, they make the most of variance
    return np.dot(eigenvectors.T, data.T - m.T).T, eigenvalues, eigenvectors, m.T  # transformed data, eigenpairs, mean


# print some properties of each cluster
def summarize(clusters, exact, k, models, maxNs):
    for i in range(k):  # for each cluster
        clust = [exact[p, :] for p in range(len(clusters)) if clusters[p] == i]
        print(f"\nCluster {i + 1}")
        # distribution of models in cluster
        print("Models:\n\t" + "\t\t ".join([str(model) for model in models]))  # distribution of models in cluster
        print("\t" + "\t\t ".join([str(len([c for c in clust if c[0] == model])) for model in models]))
        # distribution of maxN in cluster
        print("maxN:\n\t" + " \t".join([str(round(np.log10(maxN) * 100) / 100) for maxN in maxNs]))
        print("\t" + " \t\t".join([str(len([c for c in clust if c[1] == maxN])) for maxN in maxNs]))


# compute polynomial surface approximating cluster
def approximate(points, n):
    if n == 1:  # linear approximation
        X = np.concatenate([np.ones((1, points.shape[0]))[0, :], points[:, 0],
                            points[:, 1]]).reshape(-1, points.shape[0]).T
    elif n == 2:  # quadratic approximation
        X = np.concatenate([np.ones((1, points.shape[0]))[0, :], points[:, 0], points[:, 1],
                            points[:, 0] * points[:, 1], points[:, 0] ** 2,
                            points[:, 1] ** 2]).reshape(-1, points.shape[0]).T
    else:  # cubic approximation
        X = np.concatenate([np.ones((1, points.shape[0]))[0, :], points[:, 0], points[:, 1],
                            points[:, 0] * points[:, 1], points[:, 0] ** 2,
                            points[:, 1] ** 2, points[:, 0] ** 2 * points[:, 1], points[:, 0] * points[:, 1] ** 2,
                            points[:, 0] ** 3, points[:, 1] ** 3]).reshape(-1, points.shape[0]).T
    b = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, points[:, 2]))  # coefficients of surface
    text = f"\nx3 = {b[0]:.4e} + {b[1]:.4e} x1 + {b[2]:.4e} x2"  # surface coefficients as string
    if n > 1:
        text += f"\n\t   + {b[3]:.4e} x1 x2 + {b[4]:.4e} x1^2 + {b[5]:.4e} x2^2"
    if n > 2:
        text += f"\n\t   + {b[6]:.4e} x1^2 x2 + {b[7]:.4e} x1 x2^2 + {b[8]:.4e} x1^3 + {b[9]:.4e} x2^3"
    return b, text  # return coefficients and surface in string


# compute clusters, draw and print some properties
def cluster(logarithmic_scale=True, by_histograms=True, ks=None):
    if ks is None:  # ks should bi a list of k-s, that are number of clusters in k-means
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
    bins = np.linspace(-1 if logarithmic_scale else 0, 20 if logarithmic_scale else 5e10, len(eigval))

    print(f"\n\n\n{name}s:")
    plt.figure(figsize=(16, 12))
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
    plt.xlabel("Component of space")
    plt.ylabel("Explained variance [%]")
    plt.title(f"Cumulative explained variance by component for transformed {name} space")
    plt.subplot(223)
    plt.plot(np.log10(eigval + 1e-17))
    plt.xlabel("Component of space")
    plt.ylabel("Logarithm of eigenvalues by component")
    plt.title(f"Logarithm of eigenvalues for transformed {name} space")
    plt.subplot(224)
    plt.plot(t, np.abs(eigvec[:, 0]), label="1")
    plt.plot(t, np.abs(eigvec[:, 1]), label="2")
    plt.plot(t, np.abs(eigvec[:, 2]), label="3")
    plt.plot(t, np.abs(eigvec[:, 3]), label="4")
    plt.xlabel("Component of space")
    plt.ylabel("Weight")
    plt.title(f"Weights (eigenvectors) for first dimensions in transformed {name} space")
    plt.legend(loc="best")
    plt.show()

    for k in ks:  # compute k clusters, means of clusters
        kmeans = KMeans(n_clusters=k).fit(data)
        clusters = kmeans.labels_
        means = kmeans.cluster_centers_
        # column in array means is mean of cluster
        means = (np.dot(eigvec, means.T) + np.dot(meandata, np.ones((k, 1)).T))
        if name in "histogram":  # normalize means that they are distributions
            means = (means.T / np.sum(means, 0).reshape(-1, 1)).T
        # print some properties of clusters
        summarize(clusters, exact, k, models, maxNs)

        # draw means of clusters
        plt.figure(figsize=(7, 7))
        for i in range(k):
            plt.plot(t, means[:, i], label=f"{i + 1}")
        plt.xlabel("Component of space")
        plt.ylabel(f"Mean {'density' if by_histograms else 'moments'}")
        plt.title(f"Mean {name}s for each cluster")
        plt.legend(loc="best")

        # draw survival functions of means if means are distributions (1-F(t), F(t) = int_0^t f(s) ds)
        if name in "histograms":
            plt.figure(figsize=(7, 7))
            means2 = 1 - np.cumsum(means / np.sum(means, 0), 0)
            for i in range(k):
                plt.plot(t, means2[:, i], label=f"{i + 1}")
            plt.xlabel("Component of space")
            plt.ylabel(f"Probability to survive so many years")
            plt.title(f"Survival function of cluster means")
            plt.legend(loc="best")

        # list of 10 colors for at most 10 clusters and models
        colors = ["tab:" + c for c in
                  ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]]

        # draw points transformed by PCA coloured by models
        plt.figure(figsize=(16, 12))
        ax = plt.subplot(121, projection='3d')
        for model in models:
            trues = (exact[:, 0] == model)  # points of selected model
            trues = trues * np.random.random(trues.shape[0]) > 0.995  # less points to easier drawing
            for maxN in maxNs:  # set size of marker to log10(maxN)
                ax.plot(data[trues * (exact[:, 1] == maxN), 0],
                        data[trues * (exact[:, 1] == maxN), 1],
                        data[trues * (exact[:, 1] == maxN), 2],
                        marker='o', markersize=np.log10(maxN) + 1, linewidth=0, color=colors[model - 1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.title("Coloured by model - after Principal component analysis (PCA)")

        # draw clusters with approximation surface
        ax = plt.subplot(122, projection='3d')
        for i in range(k):
            points = data[clusters == i, :3]
            if name in "histograms":   # draw surfaces of order n
                n = 3
                b, text = approximate(points, n)
                print(f"cluster {i + 1} surface:\n\t" + text)
                trues = np.random.random(points.shape[0]) > 0.99   # make triangulation on less points
                x = points[trues, 0]
                y = points[trues, 1]
                Z = 1 * b[0] + x * b[1] + y * b[2]   # compute approximated x3
                if n > 1:
                    Z += x * y * b[3] + x ** 2 * b[4] + y ** 2 * b[5]
                if n > 2:
                    Z += x ** 2 * y * b[6] + x * y ** 2 * b[7] + x ** 3 * b[8] + y ** 3 * b[9]
                # Plot the surface.
                tri = Triangulation(x, y)
                ax.plot_trisurf(x, y, Z, triangles=tri.triangles, color=colors[i], linewidth=0, shade=True, alpha=0.8)
            trues = np.random.random(points.shape[0]) > 0.995    # draw less points
            ax.plot(points[trues, 0], points[trues, 1], points[trues, 2], label=i + 1, markersize=4, marker='.',
                    linewidth=0)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.legend(loc="best")
        plt.title("Coloured by cluster")

        # select 3 different histograms from convex hull
        if name in "histograms":
            
            # transform point back from pca to histogram space
            def get_histogram(point):
                p = np.dot(eigvec, point.reshape(-1, 1)) + meandata
                p = (p.T / np.sum(p, 0).reshape(-1, 1)).T
                return p / max(p)

            plt.figure(figsize=(7, 7))
            for i in range(k):
                points = data[clusters == i, :]
                hull = ConvexHull(points[:, :2])
                points = points[hull.vertices, :]  # only points from convex hull of a cluster
                m = np.mean(points[:, :2], 0)  # mean of hull
                # select 3 most different
                p1 = points[np.argmax(np.sum((points[:, :2] - m)**2, 1)), :]
                p2 = points[np.argmin(np.sum((-(p1[:2]-m)/2+(p1[:2]-m)[::-1]*np.sqrt(3)/2-points[:, :2]) ** 2, 1)), :]
                p3 = points[np.argmin(np.sum((-(p1[:2]-m)/2-(p1[:2]-m)[::-1]*np.sqrt(3)/2-points[:, :2]) ** 2, 1)), :]
                m = means[:, i]   # mean of cluster
                # draw histograms of mean and 3 others for each cluster
                plt.plot(bins, get_histogram(p1), colors[i], linewidth=1, label=f"Cluster {i+1}")
                plt.plot(bins, get_histogram(p2), colors[i], linewidth=1)
                plt.plot(bins, get_histogram(p3), colors[i], linewidth=1)
                plt.plot(bins, m / max(m), colors[i], linewidth=2)
            plt.xlabel(f"Expected civilization longevity in {'log(years)' if logarithmic_scale else 'years'}")
            plt.ylabel("Relative probability")
            plt.legend(loc="best")
            plt.title(f'Logarithmic scale distributions of models clustered by histograms' if logarithmic_scale else
                      "Linear scale distributions of models clustered")
        # show everything at once about each cluster
        plt.show()
