from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from matplotlib.tri import Triangulation
from models import *
from sklearn.decomposition import PCA


for mu, minB, maxB in [(0.5, 0, 3), (-0.5, -2, 0)]:
    plt.figure(figsize=(6, 3), dpi=200, tight_layout=True)
    plt.suptitle(f"Distributions of x from {10 ** minB} to {10 ** maxB}"
                 f" with mean (peak) at {10 ** mu:.2f}")
    plt.subplot(1, 2, 1)
    for d in distributions:
        p = sample_value((mu, minB, maxB), d, noIterations)
        h, b = np.histogram(p, 32, (minB, maxB))
        plt.plot(b[:-1], h / noIterations, label=d)
    plt.legend()
    plt.ylabel("Density")
    plt.xlabel("log(x)")
    plt.title("Logarithmic scale")
    plt.subplot(1, 2, 2)

    for d in distributions:
        h, b = np.histogram(10 ** sample_value((mu, minB, maxB), d, noIterations), 32,
                            (10 ** minB, 10 ** maxB if mu < 0 else 10 ** (maxB - 1.5)))
        plt.plot(b[:-1], h / noIterations, label=d)
    plt.legend()
    plt.xlabel("x")
    plt.title("Linear scale")
    plt.savefig(f"out/distributions_{minB}-{mu}-{maxB}.png")
    plt.show()

plt.figure(figsize=(3, 3), dpi=200, tight_layout=True)
plt.suptitle(f"Distributions of x from 1 to 1e8\nwith mean (peak) at 1e6")
for d in ["loglinear", "uniform", "loguniform"]:
    p = sample_value(bounds["N2"], d, noIterations)
    h, b = np.histogram(p, 32, (np.log10(np.min(nrange2)), np.log10(np.max(nrange2))))
    plt.plot(b[:-1], h / noIterations, label=d)
plt.legend()
plt.ylabel("Density")
plt.xlabel("log(x)")
plt.title("Logarithmic scale")
plt.savefig(f"out/distributions_bigger-N.png")
plt.show()


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
def cluster(model=0, ks=None, supermodel=1, sci_view=True):
    if not sci_view:
        plt.close('all')
    title = ["Supermodel", "Model I", "Model II", "Model III", "Model IV"][model]
    title = f"{title}{'' if model or supermodel == 1 else ' ' + str(supermodel)}"
    p = np.load(f"{collected_folder}/pca_parameters.npy")
    hists = np.load(f"{collected_folder}/pca_histograms{'' if supermodel == 1 else '_supermodel2'}.npy")
    parameters = np.array([list(m) for m in p])
    if supermodel == 2:
        selected = np.logical_not((parameters[:, 0] == 4) * (np.random.random(parameters.shape[0]) < 0.9))   # delete 90 % of 4. model
        hists = hists[selected, :]
        parameters = parameters[selected, :]

    if model:
        selected = parameters[:, 0] == model
    else:
        selected = parameters[:, 0] > 0
    hists = hists[selected, :]
    parameters = parameters[selected, :]

    super_model = np.mean(hists, 0)
    hists = np.sum(hists, 1)

    if sci_view:
        X, Y = np.meshgrid(xLogL, xLogN if supermodel == 1 else xLogN2)
        plt.figure(figsize=(5, 5), dpi=300, tight_layout=True)  # draw distributions

        ax = plt.subplot(projection='3d')
        ax.plot_surface(X, Y, super_model, cmap=plt.get_cmap("jet"),
                            shade=True, alpha=0.8)
        plt.xlabel("log(L)")
        plt.ylabel("log(N)")
        plt.suptitle(title, size=20)
        plt.title("Mean density")
        plt.savefig(f"out/{title.replace(' ', '-')}_mean-3D.png")
        plt.show()

        plt.figure(figsize=(5, 4), dpi=300, tight_layout=True)
        plt.imshow(super_model[::-1, :], cmap=plt.get_cmap("jet"),
                   extent=[xLogL.min(), xLogL.max(), (xLogN if supermodel == 1 else xLogN2).min(),
                           (xLogN if supermodel == 1 else xLogN2).max()])
        plt.xlabel("log(L)")
        plt.ylabel("log(N)")
        plt.title("Mean density - heatmap")
        plt.savefig(f"out/{title.replace(' ', '-')}_mean_heatmap.png")
        plt.show()

        plt.figure(figsize=(10, 4), dpi=300, tight_layout=True)
        plt.subplot(131)
        plt.plot(xLogL, np.sum(super_model, 0))
        plt.title("Mean density function by L")
        plt.xlabel("log(L)")
        plt.ylabel("Density")
        plt.subplot(132)
        plt.plot(xLogL, 1 - np.cumsum(np.sum(super_model, 0)))
        plt.title("Mean survival function by L")
        plt.xlabel("log(L)")
        plt.ylabel("Probability")
        plt.grid(alpha=0.4)
        plt.subplot(133)
        plt.plot(xLogN if supermodel == 1 else xLogN2, np.sum(super_model, 1))
        plt.title("Mean density function by N")
        plt.xlabel("log(N)")
        plt.ylabel("Density")
        plt.savefig(f"out/{title.replace(' ', '-')}_mean-properties.png")
        plt.show()

    colors = ["tab:" + c for c in
              ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]]
    # import data (parameters and than data)
    data0 = hists
    # make lists of parameters
    models = sorted(list({a[0] for a in parameters}))
    # exact parameters before clustering
    exact = np.array([a[0] for a in parameters])

    minimal = np.min([np.sum(exact == model) for model in models]) * 5  # we need to know how many histograms came from
    # which model that we can draw later as many as possible points but still equal number for each model

    # remove linear dependency between dimensions in data
    pca = PCA()
    pca.fit(data0.real)
    data, eigval, eigvec, meandata = pca.transform(data0), pca.explained_variance_, pca.components_.T, pca.mean_.T

    if model < 0:
        plt.figure(figsize=(10, 7), dpi=300, tight_layout=True)
        # draw how important is each dimension in transformed data (eigenvalues) and
        # how is it generated from original data (eigenvectors)
        plt.subplot(221)
        plt.plot(eigval[:7])
        plt.xticks(np.arange(0, 7), np.arange(1, 8))
        plt.xlabel("Component of space")
        plt.ylabel("Eigenvalue")
        plt.title("Eigenvalues\nfor transformed histogram space")
        plt.subplot(222)
        plt.plot(np.cumsum(eigval[:7]) / np.sum(eigval))
        plt.xticks(np.arange(0, 7), np.arange(1, 8))
        plt.xlabel("Component of space")
        plt.ylabel("Explained variance [%]")
        plt.title("Cumulative explained variance by component\nfor transformed histogram space")
        plt.subplot(223)
        plt.plot(np.log10(eigval[:7]))
        plt.xticks(np.arange(0, 7), np.arange(1, 8))
        plt.xlabel("Component of space")
        plt.ylabel("Logarithm of eigenvalues by component")
        plt.title("Logarithm of eigenvalues\nfor transformed histogram space")
        plt.subplot(224)
        plt.plot(np.abs(eigvec[:, 0]), label="1")
        plt.plot(np.abs(eigvec[:, 1]), label="2")
        plt.plot(np.abs(eigvec[:, 2]), label="3")
        plt.xlabel("Component of space")
        plt.ylabel("Weight")
        plt.title("Weights (eigenvectors) for first dimensions\nin transformed histogram space")
        plt.legend(loc="best")
        plt.show()

    for k in ks:  # compute k clusters, means of clusters
        kmeans = KMeans(n_clusters=k).fit(data.real)
        clusters = kmeans.labels_
        means = kmeans.cluster_centers_
        pca_means = kmeans.cluster_centers_
        # column in array means is mean of cluster
        means = np.dot(eigvec, means.T) + np.dot(meandata.reshape((-1, 1)), np.ones((k, 1)).T)
        means = means / np.sum(means)  # normalize means that they are distributions
        # print some properties of clusters

        if sci_view:
            plt.figure(figsize=(11, 5), dpi=300, tight_layout=True)
            plt.suptitle(f"Means of {k} clusters")
            plt.subplot(121)
            # draw means of clusters
            for i in range(k):
                med = np.power(10, xLogL)[np.argmin(np.abs(np.cumsum(means[:, i] / np.sum(means[:, i]))-0.5))]
                # avg = np.sum(xLogL*means[:, i])/np.sum(means[:, i])
                plt.plot(xLogL, means[:, i], color=colors[i], label=f"{i + 1}, median = {med:2.1f}")
            plt.xlabel("log(L)")
            plt.ylabel("Mean density")
            plt.title("Mean histogram for each cluster")
            plt.legend(loc="best")

            # draw survival functions of means if means are distributions (1-F(t), F(t) = int_0^t f(s) ds)
            plt.subplot(122)
            means2 = 1 - np.cumsum(means / np.sum(means, 0), 0)
            for i in range(k):
                plt.plot(xLogL, means2[:, i], color=colors[i], label=f"{i + 1}")
            plt.xlabel(f"log(L)")
            plt.ylabel("Probability to survive so many years")
            plt.title("Survival function of cluster means")
            plt.legend(loc="best")
            plt.grid(alpha=0.4)

            plt.savefig(f"out/{title.replace(' ', '-')}_mean-cluster-properties_{k}-clusters.png")
            plt.show()
            # list of 10
            # colors for at most 10 clusters and models

        # draw points transformed by PCA coloured by models
        plt.figure(figsize=(10, 6), dpi=300 if sci_view else 100, tight_layout=True)
        plt.suptitle(f"{title.replace(':', '')} after Principal component analysis (PCA)")
        ax = plt.subplot(121, projection='3d')
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        tr = np.array([False] * len(data))
        for m in models:  # take just random 125*9 data from selected model
            trues = np.random.choice(np.array(range(len(data)))[exact == m],
                                     size=min(np.sum(exact == m), minimal), replace=False)
            trues = np.array([True if i in trues else False for i in range(len(data))])
            tr += trues
            points = data[trues, :]  # less points to easier drawing
            ax.plot(points[:, 0],  # color by maxN
                    points[:, 1],
                    points[:, 2],
                    marker='o', markersize=4, linewidth=0, color=colors[m - 1], label=m)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.legend(loc="best")
        plt.title("Coloured by model")

        # draw clusters with approximation surface
        ax = plt.subplot(122, projection='3d')
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')
        for i in range(k):
            points = data[clusters == i, :3]
            trues = tr[clusters == i]
            n = 3
            b, text = approximate(points, n)
            # print(f"\nPloskev gruče {i + 1}:" + text if slo else f"\ncluster {i + 1} surface:" + text)
            x = points[trues, 0]
            y = points[trues, 1]
            Z = 1 * b[0] + x * b[1] + y * b[2]  # compute approximated x3
            if n > 1:
                Z += x * y * b[3] + x ** 2 * b[4] + y ** 2 * b[5]
            if n > 2:
                Z += x ** 2 * y * b[6] + x * y ** 2 * b[7] + x ** 3 * b[8] + y ** 3 * b[9]
            # Plot the surface.
            tri = Triangulation(x, y)
            ax.plot_trisurf(x, y, Z, triangles=tri.triangles, color=colors[i], linewidth=0, shade=True, alpha=0.8)
            ax.plot(points[trues, 0], points[trues, 1], points[trues, 2], color=colors[i], label=i + 1, markersize=4,
                    marker='.', linewidth=0)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        plt.legend(loc="best")
        plt.title(f"Coloured by cluster ({k} clusters)")
        plt.savefig(f"out/{title.replace(' ', '-')}_PCA_{k}-clusters.png")
        plt.show()
        if supermodel == 1 and model != 0:
            parameter_list = (["log(N)", "log(f_a)", "log(f_b)"] if model == 2 else
                          ["log(N)", "log(R_*)", "log(f_p)", "log(n_e)", "log(f_l)", "log(f_i)", "log(f_c)"] if model < 4 else
                          ["log(R_*)", "log(n_e)", "log(N_* n_e)", "log(f_g)", "log(f_pm)", "log(f_m)"])
            plt.figure(figsize=(10, 5), dpi=300 if sci_view else 100, tight_layout=True)
            plt.suptitle(f"Percent of submodels in model {model} distributed with\n"
                         f"particular distribution on particular parameter in each of\n"
                         f"{k} clusters")
            columns = np.array([[7, 1, 2, 3, 4, 5, 6], [7, 2, 6], [7, 1, 2, 3, 4, 5, 6], [2, 4, 3, 5, 6, 7]][model - 1])
            for i in range(k):
                cluster_parameters = parameters[clusters == i, :][:, columns].T
                pojavitve = np.array([np.histogram(p, 3, (-1 / 2, 2 + 1 / 2))[0] / cluster_parameters.shape[1] for ip, p in
                                      enumerate(cluster_parameters)])
                plt.subplot(1, k, i+1)
                plt.imshow(pojavitve, vmin=0, vmax=1)
                for ix in range(cluster_parameters.shape[0]):
                    for iy in range(len(distributions)):
                        plt.text(iy, ix, f"{round(pojavitve[ix, iy]*100)}%",
                                 fontsize=12, color="black" if pojavitve[ix, iy] > 0.49 else "white",
                                 va='center', ha='center')
                if i == 0:
                    plt.ylabel("Parameter")
                    plt.yticks(np.arange(0, len(parameter_list)), parameter_list)
                else:
                    plt.yticks([])
                plt.xticks(np.arange(0, 3), distributions, rotation=90)
                plt.xlabel("Distribution")
                plt.title(f"cluster {i + 1}, size={cluster_parameters.shape[1]}")

            plt.savefig(f"out/{title.replace(' ', '-')}_importance_{k}-clusters.png")
            plt.show()

            means = [tuple(p) for p in pca_means[:, :3]]
            order = sorted(list(range(len(means))), key=lambda x: means[x])
            parameter_list = (["log(f_b)", "log(f_a)", "log(N)"] if model == 2 else
                              ["log(N)", "log(f_p)", "log(f_c)"] if model < 4 else
                              ["log(N_* n_e)", "log(f_pm)", "log(f_m)"])
            columns = np.array([7, 3, 1] if model == 2 else [1, 3, 7] if model < 4 else [3, 6, 7])
            plt.figure(figsize=(7, 5), dpi=300 if sci_view else 100, tight_layout=True)
            tuples = sorted(list({tuple(t) for t in parameters[:, columns]}))
            cluster_parameters = np.array([[[tuple(t) for t in parameters[clusters == i, :][:, columns]].count(t)
                                            for t in tuples] for i in order])
            cluster_parameters = np.cumsum(cluster_parameters, 0)
            tuples = sorted(tuples, key=lambda x: list(cluster_parameters[:, tuples.index(x)]))
            cluster_parameters = np.array(sorted([list(cluster_parameters[:, c]) for c in range(cluster_parameters.shape[1])]))
            for idx, i in enumerate(reversed(order)):
                plt.bar(np.arange(0, len(tuples)), cluster_parameters[:, k-idx-1], color="white", alpha=1)
                plt.bar(np.arange(0, len(tuples)), cluster_parameters[:, k-idx-1], color=colors[i],
                        label=i+1, alpha=0.5)
            plt.xticks(list(range(len(tuples))), tuples, rotation=90)
            plt.legend(loc="best")
            plt.title(f"Percentage of distributions\n(${'$, $'.join(parameter_list)}$) by ({', '.join(distributions)})"
                      f"\ncoloured by cluster")

            plt.savefig(f"out/{title.replace(' ', '-')}_cluster-distributions_{k}-clusters.png")
            plt.show()
