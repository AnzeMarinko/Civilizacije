# CIVILISATIONS

* Author of code: Anže Marinko
* Date: February 2020 - January 2022
* Department of intelligent systems - Jožef Stefan Institute (E9 JSI)

## Instalation

Python 3 is needed for running. We recommend
using virtual environment.
Then we install all required libraries 
and generate data (It could take few hours!)
using terminal:
```
pip install -r requirements.txt
python generateData_L.py
```

## Report generating

We run all the cells in Jupyter notebook: 
`supermodel.ipynb`. It could take few minutes.

We can convert it after execution 
to `.pdf` format running command:
```
jupyter nbconvert --to pdf supermodel.ipynb --output out/supermodel.ipynb
```

All the output files are now in folder: `out`.

## Rule inspector

Run command
```
bokeh serve interactive_graphs.py
```
and open URL address printed in terminal
to open program for rule visualisation.
Abort server when finished using program
pressing Ctrl+C.
