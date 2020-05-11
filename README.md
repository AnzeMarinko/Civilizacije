# CIVILISATIONS

* Authors: *Anže Marinko, Klara Golob, Ema Jemec*
* Mentor: *dr. Matjaž Gams*
* Institution: Jožef Stefan Institute - Ljubljana, Slovenia

We want to estimate distribution of time that some civilisation
survives in our galaxy.

All code is made in Python3.

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
planets to first model.

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

Parameters to be set:

* scale of *x* axis for histograms `logarithmic_scale`,
selected model `draw_model` and distribution to draw
in 3D `draw_distribution` that is tuple of 6 (3 at model 2)
integers from 0 to 4 that represent each one random variable
in model distributed in selected distribution
* list of numbers of clusters to make `ks` from 1 to 10 so it
makes an analysis of data clustered in each number of clusters
from the list

## Report

There is going to be a report in slovenian language in
`porocilo-civilizacije-IJS_maj2020.pdf`.

## TODO:

* model 4 - mean histogram of models 1 and 3 at same
   distribution and maxN and model 2 at specific parameters
   where parameter 2 -> 1, (0 ali 1) -> 0, (3 ali 4 ali 5) -> 2,
   torej 6 moznosti za vsako kombinacijo parametrov
* model 5 - za vsake parametre iz modela 2 povprečje
   z vsemi 6 iz drugih dveh
   * modela 4 in 5 prineseta približno trikrat več novih točk
   (skupno imamo že cca. 1 120 000 točk)
   * če je to slabo, se kakega od obeh modelov znebi
* dodaj Rare Earth theory


* izračunavanje na npr. 5e6, N bolj na gosto (2**i i=0,...,14) z močnejšo mašino
* odstrani neprimerne porazdelitve
   (stabilizacija do L < 1e7, po tem odreži)
* koda, README, drugo poročilo vsega novega, zamenjaj projekt
 