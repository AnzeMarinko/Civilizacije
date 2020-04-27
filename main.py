# import all defined functions
from generateData_L import *
from plot_L import *
from plot3D_L import *
from compareHist import *
from cluster import *

# ============ Parameters ================

run_steps = [5]     # list of steps we want to run from 1 to 5
# 1: generating data, 2: drawing selected histograms, 3: drawing selected model histograms in 3D
# 4: drawing distance matrices for some different distances, 5: clustering and cluster analysis

# generating data parameters:
noIterations = 1e2    # number of generated points for each selected parameters
fixed_n = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]   # list of values for maxN

# general parameters for working with histograms
bin_no = 64    # number of bins in histograms
logarithmic_scale = True    # scale of x axis for histograms

# selected histograms for drawing from clusters made by histogram or moment comparison
by_histograms = True

# selected model and distribution to draw in 3D
# models: 1, 2, 3
draw_model = 1
# distributions: "loguniform", "uniform", "halfgauss", "lognormal", "fixed"
draw_distribution = (0, 0, 0, 0, 0, 0)

# number of clusters made
ks = [4, 10]    # number of clusters

# ============ Run functions using these parameters ============

# run only selected steps
if 1 in run_steps:
    print("\n\t1: generating data")
    generate()
if 2 in run_steps:
    print("\n\t2: draw selected histograms")
    draw_histograms(logarithmic_scale, by_histograms, bin_no)
if 3 in run_steps:
    print("\n\t3: draw histograms for selected model")
    draw_histograms3D(logarithmic_scale, draw_model, draw_distribution, bin_no)
if 4 in run_steps:
    print("\n\t4: draw distance matrices")
    distance_matrices(logarithmic_scale, bin_no)
if 5 in run_steps:
    print("\n\t5: clustering and analysis of clusters")
    print("clustering by histograms")
    cluster(logarithmic_scale, by_histograms=True, ks=[4, 7, 10])
    # + ... fixed, Y ... halfgauss, ^ ... lognormal,
    # * ... loguniform, o ... uniform
    # blue ... model 1, orange ... model 2, green ... model 3
    # size of marker ... log(N)
