"""
Anže Marinko, januar 2022

Run:
bokeh serve interactive_graphs.py
-> Open printed URL
"""
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Select, Div, Band, RadioGroup
from bokeh.plotting import figure
from tabela_napake_p import podatki, labels_list
from bokeh.palettes import *
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import itertools

data = {(label[0], subdata): (X, columns, Y) for X, columns, Y, label, subdata in podatki}
sides = ["<=", ">"]
bs = '\\'
labels = labels_list
models = sorted(list(set([s for _, s in data.keys()])))

# Set up widgets
latex_rule = Div(text=r"Rule in LaTeX")
label_select = Select(title="Label", value=labels[0], options=labels)
model_select = Select(title="Model", value=models[0], options=models)
rule_len = Select(title="Rule length", value="0", options=["0", "1", "2", "3", "4"])
rule_parts = [(Select(title=f"Rule {r + 1} column", value="N", options=["N"]),
               RadioGroup(labels=sides, active=0),
               Slider(title=f"Rule {r + 1} threshold", value=0.5, start=0.0, end=3.0, step=0.1)) for r in range(5)]

# Set up plots
line = ColumnDataSource(data=dict(x=[], y=[]))   # krivulja, ki prikazuje porazdelitev primerov
# pca točk v opazovanih dimenzijah, če je teh dimenzij dovolj, sicer še PCA iz ostalih dimenzij
points = ColumnDataSource(data=dict(x=[], y=[], color=[]))
# območje, med skrajnimi točkami preslikanimi s PCA (konveksna ovojnica teh)
areaUp = ColumnDataSource(data=dict(x=[], lower=[], upper=[]))
areaDown = ColumnDataSource(data=dict(x=[], lower=[], upper=[]))
colors = Turbo256[10:-10]
n_colors = len(colors)
plot = figure(height=600, width=700, title="Rule visualization",
              tools="crosshair,pan,reset,save,wheel_zoom,box_select,lasso_select,poly_select,tap",
              x_range=[0, 1], y_range=[0, 1])
plot.circle('x', 'y', source=points, size=7, color="color", line_color=None, fill_alpha=0.8)
plot.line('x', 'y', source=line, line_width=3, line_alpha=0.6)
plot.add_layout(Band(base='x', lower='lower', upper='upper', source=areaUp, fill_color="olivedrab",
                     level='underlay', fill_alpha=0.2, line_width=0))
plot.add_layout(Band(base='x', lower='lower', upper='upper', source=areaDown, fill_color="olivedrab",
                     level='underlay', fill_alpha=0.2, line_width=0))


class Properties:
    def __init__(self):
        self.label, self.model, self.X, self.columns, self.Y, self.minimal, self.mean, self.maximal = (None,) * 8
        self.bounds_all, self.selection, self.inds, self.not_inds, self.pca, self.data, self.color = (None,) * 7
        self.mx, self.dx, self.my, self.dy = (None,) * 4

    def rule_values(self, rule, prob):
        trues = self.Y > -10
        for _, col, side, thresh in rule:
            if ">" in side:
                trues = (self.X[:, self.columns.index(col)] > thresh) * trues
            else:
                trues = (self.X[:, self.columns.index(col)] <= thresh) * trues
        if np.sum(trues) == 0:
            return 0, 0, 0, trues
        return round(np.mean(self.Y[trues]) * 100 if prob else 10 ** np.mean(self.Y[trues]), 0 if prob else 1), round(
            np.mean(trues) * 100), round(np.std(self.Y[trues]) ** 2, 3)

    def process_rule(self, rule):
        rule_text = [] if len(rule) else ["(T)"]
        for _, col, side, thresh in rule:
            if "Model" in col:
                rule_text.append(f"({'not~' if '<=' in side else ''}{col.replace(' ', '~')})")
                continue
            tresh = 10 ** round(thresh, 1)
            prob = "f_" in col and "f_a" not in col
            tresh = round(tresh * (100 if prob else 1), round(1 - thresh) - (2 if prob else 0)) \
                if "N_*" not in col else f"{round(tresh / 1e10, 1)} * " + "10^{10}"
            rule_text.append(f"({col} {side.replace('<=', bs + 'leq')} {tresh}{'~' + bs + '%' if prob else ''})")
        rule_text = " \\land ".join(rule_text)
        val, siz, mse = self.rule_values(rule, "P" in self.label)
        rule_text = f"${rule_text} \\Rightarrow {self.label} = {val}{'~' + bs + '%' if 'P' in self.label else ''}$," \
                    f"\\hfill Size={siz} \\%, MSE={mse}"
        rule = rule_text.split('$')[1]
        tex = "<big>Rule in LaTeX:</big>"
        return fr"<h1>{self.model}</h1><h3>$${rule}$$</h3><h2>Size = {siz} %, MSE = {mse}</h2>{tex}</p>{rule_text}"

    def update(self, attrname, old, new):
        # izbor pogleda in modela
        self.label = label_select.value
        self.model = model_select.value
        # nastavitve za izbor
        self.X, self.columns, self.Y = data[(self.label, self.model)]
        self.minimal = np.round(np.min(self.X, 0) - 0.05, 1)
        self.mean = np.round(np.mean(self.X, 0), 1)
        self.maximal = np.round(np.max(self.X, 0) + 0.05, 1)
        self.bounds_all = np.array([[[self.minimal[i], self.maximal[i]][c] for i, c in enumerate(comb)]
                                    for comb in itertools.product([0, 1], repeat=len(self.columns))])
        self.color = [colors[int(i)] for i in np.round((self.Y - np.min(self.Y)) / (np.max(self.Y) - np.min(self.Y)) * (n_colors - 1))]
        rule_len.value = "0"
        self.selection = []
        for i, (r0, r1, r2) in enumerate(rule_parts):
            j = min(i + 1 if "Super" in self.model else i, len(self.columns) - 1)
            r0.value = self.columns[j]
            r0.options = self.columns
            r0.visible = False
            r1.active = 0
            r1.visible = False
            r2.value = self.mean[j]
            r2.start = self.minimal[j]
            r2.end = self.maximal[j]
            r2.visible = False
        self.update_part(None, None, None)

    def update_part(self, attrname, old, new):
        new_selection = [rule_parts[i][0].value for i in range(int(rule_len.value))]
        diff = [s is v for s, v in zip(self.selection + [""] * 4, new_selection)]
        self.selection = new_selection
        # pca točk v opazovanih dimenzijah, če je teh dimenzij dovolj, sicer še PCA iz ostalih dimenzij
        self.inds = sorted(list(set([self.columns.index(i) for i in self.selection])))
        self.not_inds = [i for i in range(len(self.columns)) if i not in self.inds]
        if len(self.inds) == 0:
            self.pca = PCA(n_components=2)
            self.pca.fit(self.X[:, self.not_inds])
            self.data = self.pca.transform(self.X[:, self.not_inds])
            bound_all = self.pca.transform(self.bounds_all[:, self.not_inds])
        elif len(self.inds) >= 2:
            self.pca = PCA(n_components=2)
            self.pca.fit(self.X[:, self.inds])
            self.data = self.pca.transform(self.X[:, self.inds])
            bound_all = self.pca.transform(self.bounds_all[:, self.inds])
        else:
            self.pca = PCA(n_components=1)
            self.pca.fit(self.X[:, self.not_inds])
            self.data = np.concatenate([self.X[:, self.inds], self.pca.transform(self.X[:, self.not_inds])], 1)
            bound_all = np.concatenate([self.bounds_all[:, self.inds],
                                        self.pca.transform(self.bounds_all[:, self.not_inds])], 1)
        self.mx, self.dx = np.min(bound_all[:, 0]), np.max(bound_all[:, 0]) - np.min(bound_all[:, 0])
        self.my, self.dy = np.min(bound_all[:, 1]), np.max(bound_all[:, 1]) - np.min(bound_all[:, 1])
        points.data = dict(x=(self.data[:, 0] - self.mx) / self.dx,
                           y=(self.data[:, 1] - self.my) / self.dy, color=self.color)
        for i in range(5):
            rule_parts[i][0].visible = i < int(rule_len.value)
            rule_parts[i][1].visible = i < int(rule_len.value)
            rule_parts[i][2].visible = i < int(rule_len.value)
            if i < int(rule_len.value) and not diff[i]:
                rule_parts[i][2].value = self.mean[self.columns.index(rule_parts[i][0].value)]
                rule_parts[i][2].start = self.minimal[self.columns.index(rule_parts[i][0].value)]
                rule_parts[i][2].end = self.maximal[self.columns.index(rule_parts[i][0].value)]
        self.frule(None, None, None)

    def frule(self, attrname, old, new):
        rule = []
        for i in range(int(rule_len.value)):
            col, sid, tresh = rule_parts[i][0].value, sides[rule_parts[i][1].active], rule_parts[i][2].value
            rule.append((0, col, sid, tresh))
        latex_rule.text = self.process_rule(rule)
        rule = [(self.columns.index(c), sides.index(s), t) for _, c, s, t in rule]
        # pca točk v opazovanih dimenzijah, če je teh dimenzij dovolj, sicer še PCA iz ostalih dimenzij
        cube = [[self.minimal[i], self.maximal[i]] for i in range(len(self.columns))]
        for c, s, t in rule:
            cube[c] = [cube[c][0], max(t, cube[c][0])] if s == 0 else [min(t, cube[c][1]), cube[c][1]]
        bounds = np.array([[i[c] for i, c in zip(cube, comb)]
                           for comb in itertools.product([0, 1], repeat=len(self.columns))])
        if len(self.inds) == 0:
            bound = self.pca.transform(bounds[:, self.not_inds])
        elif len(self.inds) >= 2:
            bound = self.pca.transform(bounds[:, self.inds])
        else:
            bound = np.concatenate([bounds[:, self.inds], self.pca.transform(bounds[:, self.not_inds])], 1)
        bound = np.array(bound)

        # območje, med skrajnimi točkami preslikanimi s PCA (konveksna ovojnica teh)
        # compute convex hull
        try:
            hull = list(bound[ConvexHull(bound).vertices, :])
            hull = np.array(hull + hull)
            # split convex hull on upper and lower part
            b1, b2 = np.argmin(hull[:, 0]), np.argmax(hull[:, 0])
            b2 = b2 if b2 > b1 else b2 + hull.shape[0] // 2
            hullUp, hullDown = hull[b1:b2 + 1, :], hull[b2:b1 + hull.shape[0] // 2 + 1, :]
            if np.max(hullUp[:, 1]) < np.max(hullDown[:, 1]):
                hullDown, hullUp = hull[b1:b2 + 1, :], hull[b2:b1 + hull.shape[0] // 2 + 1, :]
            xsDown = (hullDown[:, 0] - self.mx) / self.dx
            lowerDown = (hullDown[:, 1] - self.my) / self.dy
            MyDown, myDown = lowerDown[0], lowerDown[-1]
            dxDown, mxDown = xsDown[0] - xsDown[-1], xsDown[-1]
            upperDown = [(1 - t) * myDown + t * MyDown for t in (xsDown - mxDown) / dxDown]

            xsUp = (hullUp[:, 0] - self.mx) / self.dx
            upperUp = (hullUp[:, 1] - self.my) / self.dy
            MyUp, myUp = upperUp[0], upperUp[-1]
            dxUp, mxUp = xsUp[0] - xsUp[-1], xsUp[-1]
            lowerUp = [(1 - t) * myUp + t * MyUp for t in (xsUp - mxUp) / dxUp]

            areaUp.data = dict(x=xsUp, lower=lowerUp, upper=upperUp)
            areaDown.data = dict(x=xsDown, lower=lowerDown, upper=upperDown)
            # krivulja, ki prikazuje porazdelitev primerov
            line.data = dict(x=(hull[:hull.shape[0] // 2+1, 0] - self.mx) / self.dx, y=(hull[:hull.shape[0] // 2+1, 1] - self.my) / self.dy)
        except:
            areaUp.data = dict(x=[], lower=[], upper=[])
            areaDown.data = dict(x=[], lower=[], upper=[])
            line.data = dict(x=(bound[:, 0] - self.mx) / self.dx, y=(bound[:, 1] - self.my) / self.dy)


main_prop = Properties()
main_prop.update(None, None, None)

widgets1 = [label_select, model_select]
widgets2 = [rule_len] + [r for r, _, _ in rule_parts]
widgets3 = [r for _, r, _ in rule_parts]
widgets4 = [r for _, _, r in rule_parts]
for w in widgets1:
    w.on_change('value', main_prop.update)
for w in widgets2:
    w.on_change('value', main_prop.update_part)
for w in widgets3:
    w.on_change('active', main_prop.frule)
for w in widgets4:
    w.on_change('value', main_prop.frule)
widgets = [label_select, model_select, rule_len]
for r in rule_parts:
    widgets += list(r)

# Set up layouts and add to document
inputs = column(*widgets, width=320)
curdoc().add_root(row(inputs, column(*[plot, latex_rule]), width=1600))
curdoc().title = "Rule Inspector"
