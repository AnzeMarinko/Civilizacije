# import all defined functions
from plot3D_L import *
from cluster import *

# ============ Parameters ================

slo = True   # titles and labels in slovenian or english language

# selected model and distribution to draw in 3D
logarithmic_scale = True    # scale of x axis for histograms
draw_model = 1   # models: 1, 2, 3
draw_distribution = (0, 0, 0, 0, 0, 0)  # distributions 0-4: "loguniform", "uniform", "halfgauss", "lognormal", "fixed"

# make analysis
ks = [5, 10]    # number of clusters

# ============ Run functions using these parameters ============

print("\n\t1: draw histograms for selected model")
draw_histograms3D(logarithmic_scale, draw_model, draw_distribution, slo=slo)

print("\n\t2: clustering and analysis of clusters on logarithmic scale")
cluster(logarithmic_scale=True, ks=ks, slo=slo)
print("\n\t2: clustering and analysis of clusters on linear scale")
cluster(logarithmic_scale=False, ks=ks, slo=slo)
# size of marker ... log(N)
