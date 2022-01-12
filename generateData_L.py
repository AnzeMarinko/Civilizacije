from time import time  # to measure runtime
from os import listdir, mkdir, path  # to get list of files in directory
from models import *
from multiprocessing import Pool, freeze_support  # multi-threading
import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

d = len(distributions)
parameters = list(
    set(((0, 1), (d1, d2, d3, d4, d5, d6, 0, 0)) for d1 in range(d) for d2 in range(d) for d3 in range(d)
        for d4 in range(d) for d5 in range(d) for d6 in range(d)))
parameters += [((0, 2), (d1, d2, 0, 0, 0, 0, 0, 0)) for d1 in range(d) for d2 in range(d)]
parameters += list(set(((0, 4), (d1, d2, d3, d4, d5, d6, d7, d8)) for d1 in range(d) for d2 in range(d)
                       for d3 in range(d) for d4 in range(d) for d5 in range(d) for d6 in range(d)
                       for d7 in range(d) for d8 in range(d)))
parameters += list(
    set(((0, 3), (d1, d2, d3, d4, d5, d6, 0, 0)) for d1 in range(d) for d2 in range(d) for d3 in range(d)
        for d4 in range(d) for d5 in range(d) for d6 in range(d)))
# maximal number of new histograms:
n_histograms = len(parameters) * len(list(nrange))
if not path.exists(collected_folder):
    mkdir(collected_folder)


def generate_by_n(par):  # generate histograms for all maxN-s at selected model and distributions
    if 2 in par[0]:  # model 2 has only 3 random variables etc.
        (model, dist) = ((2,), par[1][:2])
    elif 1 in par[0]:
        (model, dist) = ((1,), par[1][:6])
    elif 3 in par[0]:
        (model, dist) = ((3,), par[1][:6])
    else:
        (model, dist) = ((4,), par[1])
    name = [f"{m}_" + "_".join([str(ps) for ps in dist]) for m in model]  # make name for file
    if path.exists(f'{data_folder}/hists-{name[0]}.npy') and path.exists(f'{data_folder}/hists-{name[-1]}.npy'):
        return 0
    t = time()
    # generate points distributed by model using selected distribution and maxN
    hists = get_point(nrange2, dist, model)
    for m in range(len(model)):
        # write all histograms and moments and list of parameters in files to get processed data faster
        np.save(f'{data_folder}/hists-{name[m]}.npy', hists[m])
    percV = round(len(listdir(data_folder)) / (d ** 2 + 2 * d ** 6 + d ** 8) * 40)
    perc = round(len(listdir(data_folder)) / (d ** 2 + 2 * d ** 6 + d ** 8) * 1000) / 10
    t0 = time() - t
    print(f"\r\t[{'#' * percV}{' ' * (40 - percV)}]\t\t Model {model[0]} generated in: {t0:.2f} s\t"
          f"\t{len(listdir(data_folder))}/{d ** 2 + 2 * d ** 6 + d ** 8} ({perc} %)    ",
          flush=True, end="")
    return time() - t


def generate():
    if not path.exists(data_folder):
        mkdir(data_folder)
    freeze_support()
    # run generator of histograms on set of free parameters
    t0 = time()
    # use 12 threads, compute cumulative runtime for all histograms
    tsum = sum(Pool(12).map(generate_by_n, parameters)) + 1e-8
    # number of random variables: model 1: 6, model 2: 3, model 3: 6
    tend = time() - t0  # runtime used after multi-threading
    if tend > 1e-1:
        print(f"\n\n\tTime used: {tend // 3600}h {tend // 60 % 60}min {tend % 60:.2f}s")
        print(f"\tMultithreading speedup: {n_histograms} histograms in {tend / tsum * 100:.2f} % of time")
        print("Generating PCA data is done.")


def collect():
    t0 = time()
    if (not os.path.exists(f"{collected_folder}/pca_histograms.npy") or
            not os.path.exists(f"{collected_folder}/pca_histograms_supermodel2.npy") or
            not os.path.exists(f"{collected_folder}/pca_parameters.npy")):
        files = [(f"{data_folder}/hists-{m}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}_{p7}_{p8}.npy"
                  if path.exists(f"{data_folder}/hists-{m}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}_{p7}_{p8}.npy") else
                  f"{data_folder}/hists-{m}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}.npy"
                  if path.exists(f"{data_folder}/hists-{m}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}.npy") else
                  f"{data_folder}/hists-{m}_{p1}_{p2}.npy", dn, m,
                  [m, p1, p2, p3, p4, p5, p6, p7, p8, dn])
                 for m in models
                 for p1 in range(len(distributions))
                 for p2 in range(len(distributions))
                 for p3 in range(len(distributions))
                 for p4 in range(len(distributions))
                 for p5 in range(len(distributions))
                 for p6 in range(len(distributions))
                 for p7 in range(len(distributions))
                 for p8 in range(len(distributions))
                 for dn in range(len(distributions))]
        hists = np.array([weight_dist(np.load(file), dn) for file, dn, m, _ in files if path.exists(file)])
        hists2 = np.array([weight_dist(np.load(file), dn, 2, m) for file, dn, m, _ in files if path.exists(file)])
        param = np.array([ps for file, _, _, ps in files if path.exists(file)])
        np.save(f"{collected_folder}/pca_histograms.npy", hists)
        np.save(f"{collected_folder}/pca_histograms_supermodel2.npy", hists2)
        np.save(f"{collected_folder}/pca_parameters.npy", param)
    tend = time() - t0  # runtime used after multi-threading
    print(f"\tTime used: {tend:.2f}s")
    print("Collecting PCA data is done.")


def meti(n):
    t0 = time()
    if (not os.path.exists(f"{collected_folder}/meti_parameters.npy") or
            not os.path.exists(f"{collected_folder}/meti_labels.npy") or
            not os.path.exists(f"{collected_folder}/meti_tabela_csv.csv")):
        parameters = ["log(f_a)", "log(R_*)", "log(f_p)", "log(f_b)", "log(n_e)", "log(f_l)", "log(f_i)", "log(f_c)",
                      "log(N_* n_e)", "log(f_g)", "log(f_pm)", "log(f_m)", "log(f_j)", "log(f_me)"]
        labels = ['log(L)']
        columns = ["Supermodel", "Model"] + parameters + ['log(N)'] + labels
        frames = []
        for _ in range(n):
            sm = np.random.randint(1, 3, 1)[0]
            point = random_point(sm == 1)
            frames.append([sm] + point[0] + [point[1]])
        df = pd.DataFrame(frames, columns=columns)
        result = pd.get_dummies(df, columns=['Model'])
        result.index = list(range(1, n + 1))
        result = result.fillna(0)
        columns = ['Supermodel', 'Model_Drake', 'Model_Simplified', 'Model_Expand', 'Model_Rare_Earth',
                   'log(N)'] + parameters + labels
        result = result[columns]
        result.to_csv(f'{collected_folder}/meti_tabela_csv.csv')
        parameters = result[['Supermodel', 'Model_Drake', 'Model_Simplified', 'Model_Expand', 'Model_Rare_Earth', 'log(N)'] + parameters]
        labels = result[labels]
        np.save(f'{collected_folder}/meti_parameters.npy', parameters)
        np.save(f'{collected_folder}/meti_labels.npy', labels)
    tend = time() - t0  # runtime used after multi-threading
    print(f"\tTime used: {tend:.2f}s")
    print("Generating ML data (shots) is done.\n")


if __name__ == "__main__":
    print("\nIt will take few hours to execute.")
    generate()
    collect()
    meti(30000)
