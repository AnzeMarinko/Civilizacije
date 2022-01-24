from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from models import *
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'

printing_trees = False

data = pd.read_csv(f'{collected_folder}/meti_tabela_csv.csv', index_col=0)
parameters = list(data.columns)[1:-1]
label_columns = [("L", lambda x: np.reshape(x, (1, -1)).T),
                 ("P(L < 1 000)", lambda x: np.reshape(np.where(x < 3, 1, 0), (1, -1)).T),
                 ("P(1 000 < L < 10 000)", lambda x: np.reshape(np.where(x < 4, np.where(x > 3, 1, 0), 0), (1, -1)).T),
                 ("P(10 000 < L < 100 000)", lambda x: np.reshape(np.where(x < 5, np.where(x > 4, 1, 0), 0), (1, -1)).T),
                 ('P(L > 100 000)', lambda x: np.reshape(np.where(x > 5, 1, 0), (1, -1)).T)]
labels_list = [l[0] for l in label_columns]
dataTotal = np.load(f'{collected_folder}/meti_parameters.npy')
labelsTotal = np.load(f'{collected_folder}/meti_labels.npy')
columns_id = [[4, 6, 7, 8, 10, 11, 12], [4, 5, 9],
              [4, 6, 7, 8, 10, 11, 12], [6, 8, 13, 14, 15, 16, 17, 18]]

data = dataTotal[dataTotal[:, 0] == 1, 1:]
labels = labelsTotal[dataTotal[:, 0] == 1]
model1 = data[:, 0] == 1
model2 = data[:, 1] == 1
model3 = data[:, 2] == 1
model4 = data[:, 3] == 1
data2 = dataTotal[dataTotal[:, 0] == 2, 1:]
labels2 = labelsTotal[dataTotal[:, 0] == 2]
selected = np.logical_not((data2[:, 3] == 1) * (np.random.random(data2.shape[0]) < 0.9))  # delete 90 % of 4. model
data2 = data2[selected, :]
labels2 = labels2[selected]
model3_2 = data2[:, 2] == 1
podatki = []
podatki += [(data[model1, :][:, columns_id[0]], [parameters[c] for c in columns_id[0]], labels[model1], 'Model I')]
podatki += [(data[model2, :][:, columns_id[1]], [parameters[c] for c in columns_id[1]], labels[model2], 'Model II')]
podatki += [(data[model3, :][:, columns_id[2]], [parameters[c] for c in columns_id[2]], labels[model3], 'Model III')]
podatki += [(data2[model3_2, :][:, columns_id[2]], [parameters[c] for c in columns_id[2]], labels2[model3_2], 'Model III - 2')]
podatki += [(data[model4, :][:, columns_id[3]], [parameters[c] for c in columns_id[3]], labels[model4], 'Model IV')]
podatki += [(data, parameters, labels, 'Supermodel')]
podatki += [(data2, parameters, labels2, 'Supermodel 2')]


def rules_from_tree(tree, columns, printing_trees=printing_trees):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value[:, 0, 0]
    n_node_samples = tree.tree_.n_node_samples
    impurity = tree.tree_.impurity

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    if printing_trees:
        print("\nThe binary tree structure has {n} nodes and has the following tree structure:".format(n=n_nodes))
        for i in range(n_nodes):
            if is_leaves[i]:
                print("{space}node={node} is a leaf node with value {value:.1f} with {n_node_samples} samples,"
                      " gini impurity = {impurity}.".format(
                    space=(node_depth[i] + 1) * "\t", node=i, value=value[i], n_node_samples=n_node_samples[i],
                    impurity=impurity[i]))
            else:
                print("{space}node={node} is a split node: go to node {left} if {feature} <= {threshold:.1f} "
                      "else to node {right}.".format(space=(node_depth[i] + 1) * "\t", node=i, left=children_left[i],
                                                     feature=columns[feature[i]], threshold=threshold[i],
                                                     right=children_right[i]))

    def recursive_rule(i=0):
        if is_leaves[i]:
            return [[]]
        col = columns[feature[i]]
        if "Model" in col:
            return [[(f"not~{col.replace(' ', '~')}", col, "<=", 0.5)] + r for r in recursive_rule(children_left[i])] + \
                   [[(col.replace(' ', '~'), col, ">", 0.5)] + r for r in recursive_rule(children_right[i])] + \
                   [[]]
        tresh = 10 ** round(threshold[i], 1)
        prob = "f_" in col and "f_a" not in col
        tresh = round(tresh * (100 if prob else 1), round(1 - threshold[i]) - (2 if prob else 0)) \
            if "N_*" not in col else f"{round(tresh / 1e10, 1)} * " + "10^{10}"
        tresh = f"{tresh}~\\%" if prob else tresh
        return [[(f"{col} \\leq {tresh}", col, "<=", threshold[i])] + r for r in recursive_rule(children_left[i])] + \
               [[(f"{col} > {tresh}", col, ">", threshold[i])] + r for r in recursive_rule(children_right[i])] + \
               [[]]

    return [tuple(r) for r in recursive_rule()]


def check_rule(rule, X, Y, columns, prob):
    trues = Y > -10
    for _, col, side, thresh in rule:
        if ">" in side:
            trues = (X[:, columns.index(col)] > thresh) * trues
        else:
            trues = (X[:, columns.index(col)] <= thresh) * trues
    if np.sum(trues) == 0:
        return 0, 0, 0
    return round(np.mean(Y[trues]) * 100 if prob else 10 ** np.mean(Y[trues]), 0 if prob else 1), round(
        np.mean(trues) * 100), round(np.std(Y[trues]) ** 2, 3)


def modeli_napake():
    size = 2
    plt.figure(figsize=(12 * size, len(label_columns) * size), dpi=300, tight_layout=True)
    pr = ""
    for k, (X, columns, Y, subdata) in enumerate(podatki):
        for kl, label in enumerate(label_columns):
            # print(f"\r {round((j + 1)/20 * 100)}%", end=" ", flush=True)
            j = k + kl * len(podatki)
            criterion = [("squared_error", "MSE"), ("absolute_error", "MAE")][0]
            model = RandomForestRegressor(criterion=criterion[0], n_estimators=300,
                                          max_depth=4, random_state=1, min_samples_leaf=0.03)  # regresor
            model = model.fit(X, label[1](Y))
            feat_importances = pd.Series(model.feature_importances_,
                                         [f"${p}$" if "Model" not in p else p for p in columns])
            plt.subplot(len(label_columns), len(podatki), j + 1)
            feat_importances.plot(kind='bar')
            if j // 7 == 0:
                plt.title(f"{subdata}")
            if j % 7 == 0:
                plt.ylabel(f"{label[0]}\nParameter importance")
            if j // 7 < len(label_columns) - 1:
                plt.xticks([])
            rules = []
            for i, tree in enumerate(model.estimators_):  # sestavi slovar in vrni nekaj pogostejših in njihovo pojavnost
                rules += rules_from_tree(tree, columns, printing_trees and i < 3)
            # print(len(rules), len(set(rules)))

            b = 0.2 if "P" in label[0] or "Super" not in subdata else 1
            land = "\\land"
            rules = [(f"({(') ' + land + ' (').join([r for r, _, _, _ in rule]) if len(rule) > 0 else 'T'})",
                      check_rule(rule, X, label[1](Y).T, columns, "P" in label[0])) for rule in rules]
            rules = [rule for rule in rules if rule[1][-1] < b and rule[1][-2] > 5]
            aux_raw = [rule for rule, _ in rules]
            rules_raw = sorted([rule for rule in set(aux_raw)], key=lambda x: - aux_raw.count(x))
            rules_raw = [rules[aux_raw.index(rule)] for rule in rules_raw]
            rules = [(rule, stat[0], stat[1], stat[2]) for rule, stat in rules_raw]

            proc = '~\\%'
            if len(rules) > 0:
                pra = "\n\\subsection{" + f"{subdata}" + "}" + f"\n{label[0]}:\n" + "\\begin{itemize}\n"
                pra += "\n".join([f"\\item ${r[0]} \\Rightarrow {label[0]} = {r[1]}{proc if 'P' in label[0] else ''}$,"
                                  f"\\hfill Size={r[2]} \\%, MSE={r[3]}" for r in rules[:20]])
                pra += "\n\\end{itemize}\n"
                # print(pra, end="")
                pr += pra
    pr = "\\documentclass[numbered]{CSL}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n" + pr + "\n\\end{document}"
    with open("01_rules.tex", "w") as f:
        f.write(pr)
    plt.savefig(f'out/importance-random_forest_model.png')
    plt.show()


def conditionN():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    data = {(tuple(columns), subdata): (X, Y) for X, columns, Y, subdata in podatki}
    latex = "\\documentclass[numbered]{CSL}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n\n"
    file = "01_table.tex"
    for columns, subdata in [(tuple(col), s) for _, col, _, s in podatki]:
        print("\n", subdata, "    (value, size, mse) at condition on N")
        sel = [0, 1, 4, round(np.max(xLogN2))] if "2" in subdata else [0, 1, 2, 3]
        res = [[check_rule([(None, "N", ">", v1)] if v2 > sel[-2] else
                           [(None, "N", "<=", v2), (None, "N", ">", v1)] if v1 > 0 else
                           [(None, "N", "<=", v2)], X, Y, columns, "P" in label)
                for v1, v2 in zip(sel[:-1], sel[1:]) if "N" in columns] +
               [check_rule([], X, Y, columns, "P" in label)]
               for X, Y, label in [(data[(columns, subdata)][0], label[1](data[(columns, subdata)][1]).T, label[0])
                                   for label in label_columns]]
        sizes = [f"{r[1]} %" for r in res[0]]
        cols = [f"lbr {v1}, {v2} rbr" for v1, v2 in zip(sel[:-1], sel[1:])
                 if "N" in columns] + [f"lbr {sel[0]}, {sel[-1]} rbr"]
        res = [sizes, [str(r[0]) for r in res[0]]] + [[f"{int(val[0])} %" for val in r] for r in res[1:]]
        df = pd.DataFrame(np.array(res), columns=cols, index=["Size", "mean(L)"] + labels_list[1:])
        print(str(df))
        text = ("\n\n\\begin{table}[ht]\n\\tabcolsep4pt\n\\processtable{"
                "" + subdata + ", values on intervals of log(N)  \\label{table:N" + subdata.replace(" ", "") + "}}{\n"
                "\\begin{tabular}{" + "l|" + "c" * (len(cols)-1) + ("|r" if len(cols) > 1 else "r") + "}\n\\hline\n"
                "\\rowcolor{Theadcolor}  & " + " & ".join(cols) + " \\\\\\hline\n",
                "\n\\end{tabular}}{\n\\begin{tablenotes}\n%\\item table note\n\\end{tablenotes}}\n\\end{table}\n\n")
        mid = " \\\\\n".join([" & ".join([lab] + res[i]) for i, lab in enumerate(["Size", "mean(L)"] + labels_list[1:])])
        latex += text[0] + mid + text[1]
    with open(file, "w") as f:
        f.write(latex.replace("<=", "\\leq").replace(" %", " \\%").replace(
            "lbr", "$\\lbrack").replace("rbr", "\\rbrack$") + "\n\n\\end{document}")


if __name__ == "__main__":
    modeli_napake()
    conditionN()
