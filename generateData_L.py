import itertools
from time import time  # to measure runtime
from os import listdir, mkdir, path  # to get list of files in directory
from models import *
from multiprocessing import Pool, freeze_support  # multi-threading
import pandas as pd

d = len(distributions)
parameters = [*itertools.product([*range(d)], repeat=6)]


def generate_by_n(par):  # generate histograms for all maxN-s at selected model and distributions
    name = "_".join([str(ps) for ps in par])  # make name for file
    if path.exists(f'{data_folder}/hists-{name}.npy'):
        return 0
    t = time()
    # generate points distributed by model using selected distribution and maxN
    hists = get_point(nrange2, par)
    np.save(f'{data_folder}/hists-{name}.npy', hists)
    ldf = len(listdir(data_folder))
    percV = round(ldf / len(parameters) * 40)
    perc = round(ldf / len(parameters) * 100, 1)
    t0 = time() - t
    print(f"\r\t[{'#' * percV}{' ' * (40 - percV)}]\t  {ldf}/{len(parameters)} ({perc} %)  "
          f"\t  Distribution {name} generated in: {t0:.2f} s     ", flush=True, end="")
    return time() - t


def generate():
    if not path.exists(data_folder):
        mkdir(data_folder)
    freeze_support()
    ldf = len(listdir(data_folder))
    percV = round(ldf / len(parameters) * 40)
    perc = round(ldf / len(parameters) * 100, 1)
    print(f"\r\t[{'#' * percV}{' ' * (40 - percV)}]\t  {ldf}/{len(parameters)} ({perc} %)     ", flush=True, end="")
    # run generator of histograms on set of free parameters
    t0 = time()
    # use 12 threads, compute cumulative runtime for all histograms
    tsum = sum(Pool(12).map(generate_by_n, parameters)) + 1e-3
    tend = time() - t0  # runtime used after multi-threading
    print(f"\n\tTime used: {tend // 3600}h {tend // 60 % 60}min {tend % 60:.2f}s")
    print(f"\tMultithreading speedup: used only {tend / tsum * 100:.2f} % of time ({tsum / tend:.1f}-times faster)")
    print("Generating PCA data is done.")


def collect():
    t0 = time()
    if (not os.path.exists(f"{collected_folder}/pca_histograms.npy") or
            not os.path.exists(f"{collected_folder}/pca_histograms_supermodel2.npy") or
            not os.path.exists(f"{collected_folder}/pca_parameters.npy")):
        files = [(f"{data_folder}/hists-{'_'.join([str(ps) for ps in par])}.npy", list(par)) for par in parameters]
        all_hists = [(np.load(file), par) for file, par in files if path.exists(file)]
        hists = np.array([weight_dist(hist, dn) for hists, _ in all_hists
                          for m, hist in enumerate(hists) for dn in range(d)])
        hists2 = np.array([weight_dist(hist, dn, False, m+1) for hists, _ in all_hists
                          for m, hist in enumerate(hists) for dn in range(d)])
        param = np.array([[m, dn] + par for _, par in all_hists for m in models for dn in range(d)])
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
        parameters = ["Supermodel", "Model I", "Model II", "Model III", "Model IV", "N",
                      "f_a", "R_*", "f_p", "n_e", "f_b", "f_l", "f_i", "f_c",
                      "N_* n_e", "f_g", "f_{pm}", "f_m", "f_j", "f_{me}"]
        labels = ['log(L)']
        columns = parameters + labels
        frames = []
        for _ in range(n):
            for point in random_point():
                frames.append(point[0] + [point[1]])
        df = pd.DataFrame(frames, columns=columns)
        df.index = list(range(1, n * 8 + 1))
        df.to_csv(f'{collected_folder}/meti_tabela_csv.csv')
        parameters = df[parameters]
        labels = df[labels]
        np.save(f'{collected_folder}/meti_parameters.npy', parameters)
        np.save(f'{collected_folder}/meti_labels.npy', labels)
    tend = time() - t0  # runtime used after multi-threading
    print(f"\tTime used: {tend:.2f}s")
    print("Generating ML data (shots) is done.\n")


if __name__ == "__main__":
    print("\nIt will take about 30 minutes to execute.")
    generate()
    collect()
    meti(5000)
