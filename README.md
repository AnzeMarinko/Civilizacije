# CIVILISATIONS

* Author of code: Anže Marinko
* Coauthors of idea: dr. Matjaž Gams, David Susič
* Date: January 2022
* Department of intelligent systems - Jožef Stefan Institute (E9 JSI)

## Installation

Python 3 is needed for running. We recommend
using virtual environment.
Then install all required libraries 
and generate data (it takes about 30 minutes)
using terminal:
```
pip install -r requirements.txt
python generateData_L.py
```

## Report generating

We run all the cells in Jupyter notebook: 
`supermodel.ipynb`. It could take few minutes.

We can convert it after execution of all cells
to `.pdf` format running command:
```
jupyter nbconvert --to pdf supermodel.ipynb --output 01_supermodel
```

All the output files are now in:
* folder `out`,
* `01_table.tex`,
* `01_rules.tex` and
* `01_supermodel.pdf`.

## Rule inspector

Run command
```
bokeh serve interactive_graphs.py
```
and open URL address printed in terminal
to open program for rule visualisation.
Abort server when finished using program
pressing `Ctrl+C`.

## TODO:
* dodaj končno verzijo članka na Github
