# import all defined functions
from plot3D_L import *
from cluster import *

# ============ Parameters ================

# selected model and distribution to draw in 3D
logarithmic_scale = True    # scale of x axis for histograms
draw_model = 1   # models: 1, 2, 3
draw_distribution = (0, 0, 0, 0, 0, 0)  # distributions 0-4: "loguniform", "uniform", "halfgauss", "lognormal", "fixed"

# make analysis
ks = [4, 7, 10]    # number of clusters

# ============ Run functions using these parameters ============

print("\n\t1: draw histograms for selected model")
draw_histograms3D(logarithmic_scale, draw_model, draw_distribution)

print("\n\t2: clustering and analysis of clusters on logarithmic scale")
cluster(logarithmic_scale=True, ks=ks)
print("\n\t2: clustering and analysis of clusters on liner scale")
cluster(logarithmic_scale=False, ks=ks)
# blue ... model 1, orange ... model 2, green ... model 3
# size of marker ... log(N)
