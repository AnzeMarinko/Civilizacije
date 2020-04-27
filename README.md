# CIVILISATIONS

We want to estimate distribution of time that some civilisation
survives in our galaxy. There is file `main.py` explained on the
bottom in section *Running experiments* to run everything you want without knowing other code. All
code is made in Python3.

## Generate new distributions

At first you have to run `generateData_L.py` to generate data.
Generated data is than saved in logarithmic scale to directory
`data`. *Be careful: It overwrites old files!!!*

We are generating data by different models that are
defined in `models.py`. Each model takes two more parameters:
* _maxN_ ... maximal number of civilisations in our galaxy. We assume
that there is at least one civilisation (here we are) and at most 10^5
civilisations otherwise we would already see them.
* _distribution_ ... some parameters of the model are only bounded
and we have to define distribution on this interval. Possible distributions are:
_"loguniform", "uniform", "halfgauss", "lognormal", "fixed"_.

### Models

* *Model 1:* The most elementary model using Sandberg distribution,
* *Model 2:* We collect together some variables from first model so
we have only two random variables,
* *Model 3:* We add to first model a possibility of spreading to other
planets.

### Save collected data for further analysis

After generating data we make a new tables with all data prepared
for analysis:
* `parameters.txt` and `lin-parameters.txt` with list of parameters
for logarithmic or linear scale histograms and moments
* `hists.txt` and `lin-hists.txt` with all linearly or logarithmically
collected histograms for each of parameters
* `moments.txt` and `lin-moments.txt` with all linearly or logarithmically
collected histograms transformed to moment space for each of parameters

Moments are central mathematical moments. First moment is mean and
second moment is variance. We can make histograms on linear or
logarithmic scale.

## Data direct comparison

We can look at distributions and compare them pairwise.

### Drawing of selected distributions

We can draw generated data using `plot_L.py` or `plot3D_L.py` where
we have to set scale (logarithmic scale or not) and list of models
to draw (`parameters`).

### Compare histograms

Using `compareHist.py` we draw distance matrix for all generated data.
It computes different matrices, for each defined comparing method
(distance between histograms) that is defined in `compareMethods.py`.

## Clustering of generated distributions

`cluster.py`

```
+ ... fixed
Y ... halfgauss
^ ... lognormal
* ... loguniform
o ... uniform

blue ... model 1
orange ... model 2
green ... model 3

size of marker ... log(N)
```

## Running experiments

There is a file `main.py` with all commands collected.
Set parameters in first section and run the file.
Parameters to be set:

* list of steps we want to run from 1 to 5:
```
run_steps = [1, 2, 3, 4, 5]
# 1: generating data,
# 2: drawing selected histograms,
# 3: drawing selected model histograms in 3D
# 4: drawing distance matrices for some different distances,
# 5: clustering and cluster analysis
```
* generating data parameters:
```
# number of generated points for each selected parameters
noIterations = 1e6

# list of values for maxN
fixed_n = [3**i for i in range(12)]
```
* general parameters for working with histograms:
```
# number of bins in histograms
bin_no = 100

# scale of x axis for histograms
logarithmic_scale = True
```
* draw selected histograms:
```
# selected histograms from clusters made by
# histogram or moment comparison
by_histograms = True
```
* selected model and distribution to draw in 3D:
```
# models from 1 to 3
draw_model = 1

# distributions:
# "loguniform", "uniform", "halfgauss", "lognormal", "fixed"
draw_distribution = "loguniform"
```
* number of clusters made:
```
k = 4    # number of clusters
```

## Report

There is going to be a report in slovenian language in
`civilizacije.pdf`.

## TODO:

* multithreading, več iteracij (1e6)?
* poženi generiranje na močnejšem računalniku
* uredi kodo cluster.py, grafe, vso kodo prilagodi novim poimenovanjem podatkov,
uporabljaj hists.txt in ne več dejanskih podatkov
* odstrani neprimerne porazdelitve (maxN < 1e4,
stabilizacija do L < 1e7, po tem odreži)
* izračunaj pca in izriši le uniformno porazdeljenih
500 točk na gručo
* dodaj 4. model in Rare Earth theory
* uredi kodo, dokončaj README in nalozi na GitHub
* napisi poročilo vsega do sedaj v slovenščini
(kar je v README in opažanja, predvidevanja
 ob slikah, predlogi za naprej)

## Opažanja
* lognormal in loguniform ter uniform in halfgauss podobna
oz. podobno porazdeljena po grucah
* fixed obicajno popolnma zase
* prvi in tretji model podobna
