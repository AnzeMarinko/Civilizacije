# import all defined functions
from plot3D_L import *
from cluster import *

# ============ Parameters ================

run_steps = [1, 2]     # list of steps we want to run from 1 to 2
# 1: drawing selected model histograms in 3D,
# 2: clustering and cluster analysis

# general parameters for working with histograms
logarithmic_scale = True    # scale of x axis for histograms

# selected model and distribution to draw in 3D
# models: 1, 2, 3
draw_model = 3
# distributions 0-4: "loguniform", "uniform", "halfgauss", "lognormal", "fixed"
draw_distribution = (0, 0, 0, 0, 0, 0)

# make analysis
by_histograms = True   # cluster by histograms
ks = [4, 7, 10]    # number of clusters

# ============ Run functions using these parameters ============

# run only selected steps
if 1 in run_steps:
    print("\n\t1: draw histograms for selected model")
    draw_histograms3D(logarithmic_scale, draw_model, draw_distribution)
if 2 in run_steps:
    print("\n\t2: clustering and analysis of clusters")
    print(f"clustering by {'histograms' if by_histograms else 'moments'}")
    cluster(logarithmic_scale, by_histograms=by_histograms, ks=ks)
    # blue ... model 1, orange ... model 2, green ... model 3
    # size of marker ... log(N)
