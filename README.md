# CIVILISATIONS

* Authors: *Anže Marinko, Klara Golob, Ema Jemec*
* Mentor: *dr. Matjaž Gams*
* Institution: Jožef Stefan Institute - Ljubljana, Slovenia

We want to estimate distribution of time that some civilisation
survives in our galaxy.

All code is made in Python3.

## Assumptions:
* There is at least one civilisation (here we are) and at most 10 000 civilisations, otherwise we would
 already detect any of them,
* Intelligent civilisation survives at most 10^8 years and already this bound seems to be too optimistic,
* Drake equation holds and its parameters are independently distributed by some selected distributions.

## Generate new distributions

At first you have to run `python generateData_L.py` in console to generate data.
Generated data is than saved in logarithmic scale to directory
`data` than program collects all data in just few files in directory `collectedData`.

We are generating data by different models that are
defined in `models.py`. Each model takes two more parameters:
* **maxN** ... maximal number of civilisations in our galaxy. We assume
that there is at least one civilisation (here we are) and at most 10^4
civilisations otherwise we would already see them.
* **distribution** ... some parameters of the model are only bounded
and we have to define distributions in this bounds. Possible distributions are:
_"loguniform", "uniform", "halfgauss", "lognormal", "fixed"_.

#### Models

* **Model 1:** The most elementary model using Sandberg distribution explained
in `Sandberg-original-paper.pdf`, but we take parameters as random variables to
get more general model,
* **Model 2:** We collect together some variables from first model so
we have only two random variables,
* **Model 3:** We add a possibility of spreading to other
planets for the first model,
* **Model 4:** Rare Earth Theory equation included,
* **Model 5:** Interpolation of previous models to fill holes
(take combination of two random generated histograms).

#### Save collected data for further analysis

After generating data we make a new tables with all data prepared
for analysis:
* `parameters.txt` and `lin-parameters.txt` with list of parameters
for logarithmic or linear scale histograms and moments
* `hists.txt` and `lin-hists.txt` with all linearly or logarithmically
collected histograms for each of parameters

## Data comparison: Drawing of selected distributions

We can draw generated data using `plot3D_L.py` where
we have to set scale (logarithmic scale or not) and models
to draw.

## Clustering of generated distributions

We can cluster all histograms in selected number of clusters
using `cluster.py`. It draws us some plots so we can understand
generated data better. In cloud of points colored by model
size of marker is equal to `log(maxN)`.

## Running experiments

There is a file `main.py` with some commands collected
so we do not need to understand functions to draw plots.
You just have to set parameters in first section of code
and run the file.

Parameter to be set:

* list of numbers of clusters to make `ks` from 1 to 10 so it
makes an analysis of data clustered in each number of clusters
from the list
