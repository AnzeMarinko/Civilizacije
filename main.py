# import all defined functions
from cluster import *

# ============ Parameters ================

slo = True   # titles and labels in slovenian or english language
# make analysis
ks = [5, 10]    # number of clusters

# ============ Run functions using these parameters ============

print("\n\t10: clustering and analysis of clusters on logarithmic scale")
cluster(folder=10, ks=ks, slo=slo)
print("\n\t100: clustering and analysis of clusters on logarithmic scale")
cluster(folder=100, ks=ks, slo=slo)
print("\n\t1000: clustering and analysis of clusters on logarithmic scale")
cluster(folder=1000, ks=ks, slo=slo)
print("\n\t10000: clustering and analysis of clusters on linear scale")
cluster(folder=10000, ks=ks, slo=slo)
# size of marker ... log(N)
