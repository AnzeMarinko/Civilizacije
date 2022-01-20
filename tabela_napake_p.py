from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from models import *
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'

printing_trees = False

data = pd.read_csv(f'{collected_folder}/meti_tabela_csv.csv', index_col=0)
parameters = list(data.columns)[1:-1]
label_columns = [("L", lambda x: x),
                 ("P(L < 1 000)", lambda x: np.where(x < 3, 1, 0)),
                 ("P(1 000 < L < 10 000)", lambda x: np.where(x < 4, np.where(x > 3, 1, 0), 0)),
                 ("P(10 000 < L < 100 000)", lambda x: np.where(x < 5, np.where(x > 4, 1, 0), 0)),
                 ('P(L > 100 000)', lambda x: np.where(x > 5, 1, 0))]
labels_list = [l[0] for l in label_columns]
dataTotal = np.load(f'{collected_folder}/meti_parameters.npy')
labelsTotal = np.load(f'{collected_folder}/meti_labels.npy')
labelsTotal = np.concatenate([l[1](labelsTotal) for l in label_columns], 1)
columns_id = [[4, 6, 7, 8, 10, 11, 12], [4, 5, 9],
              [4, 6, 7, 8, 10, 11, 12], [6, 8, 13, 14, 15, 16, 17, 18]]

data = dataTotal[dataTotal[:, 0] == 1, 1:]
labels = labelsTotal[dataTotal[:, 0] == 1, :]
model1 = data[:, 0] == 1
model2 = data[:, 1] == 1
model3 = data[:, 2] == 1
model4 = data[:, 3] == 1
data2 = dataTotal[dataTotal[:, 0] == 2, 1:]
labels2 = labelsTotal[dataTotal[:, 0] == 2, :]
selected = np.logical_not((data2[:, 3] == 1) * (np.random.random(data2.shape[0]) < 0.9))  # delete 90 % of 4. model
data2 = data2[selected, :]
labels2 = labels2[selected, :]
model3_2 = data2[:, 2] == 1
podatki = [[], [], [], [], [], [], []]
for stolpec in range(len(label_columns)):
    podatki[0] += [(data[model1, :][:, columns_id[0]], [parameters[c] for c in columns_id[0]],
                    labels[:, stolpec][model1], label_columns[stolpec], 'Model I')]
    podatki[1] += [(data[model2, :][:, columns_id[1]], [parameters[c] for c in columns_id[1]],
                    labels[:, stolpec][model2], label_columns[stolpec], 'Model II')]
    podatki[2] += [(data[model3, :][:, columns_id[2]], [parameters[c] for c in columns_id[2]],
                    labels[:, stolpec][model3], label_columns[stolpec], 'Model III')]
    podatki[3] += [(data2[model3_2, :][:, columns_id[2]], [parameters[c] for c in columns_id[2]],
                    labels2[:, stolpec][model3_2], label_columns[stolpec], 'Model III - 2')]
    podatki[4] += [(data[model4, :][:, columns_id[3]], [parameters[c] for c in columns_id[3]],
                    labels[:, stolpec][model4], label_columns[stolpec], 'Model IV')]
    podatki[5] += [(data, parameters, labels[:, stolpec], label_columns[stolpec], 'Supermodel')]
    podatki[6] += [(data2, parameters, labels2[:, stolpec], label_columns[stolpec], 'Supermodel 2')]
podatki = podatki[0] + podatki[1] + podatki[2] + podatki[3] + podatki[4] + podatki[5] + podatki[6]


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
        return [[(f"{col} \\leq {tresh}", col, "<=", threshold[i])] + r for r in recursive_rule(children_left[i])] +\
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
    for k, (X, columns, Y, label, subdata) in enumerate(podatki):
        # print(f"\r {round((j + 1)/20 * 100)}%", end=" ", flush=True)
        j = k // len(label_columns) + (k % len(label_columns)) * 7
        criterion = [("squared_error", "MSE"), ("absolute_error", "MAE")][0]
        model = RandomForestRegressor(criterion=criterion[0], n_estimators=300,
                                      max_depth=4, random_state=1, min_samples_leaf=0.03)  # regresor
        model = model.fit(X, Y)
        feat_importances = pd.Series(model.feature_importances_,
                                     [f"${p}$" if "(" in p else p for p in columns])
        plt.subplot(len(label_columns), 7, j + 1)
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
                  check_rule(rule, X, Y, columns, "P" in label[0])) for rule in rules]
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
    data = {(label[0], tuple(columns), subdata): (X, Y) for X, columns, Y, label, subdata in podatki}
    latex = "\\documentclass[numbered]{CSL}\n\\usepackage[utf8]{inputenc}\n" \
            "\\usepackage{booktabs}\n\\begin{document}\n\n"
    file = "01_table.tex"
    for columns, subdata in sorted(list(set([(col, s) for _, col, s in data.keys()]))):
        print("\n", subdata, "    (value, size, mse) at condition on N")
        res = [[check_rule([(None, "N", "<=", value)], X, Y, columns, "P" in label)
                for value in ([1, 2, 3] if "2" in subdata else [1, 2]) if "N" in columns] +
               [check_rule([], X, Y, columns, "P" in label)]
               for (X, Y), label in [(data[(label, columns, subdata)], label) for label in labels_list]]
        df = pd.DataFrame(res, columns=[f"N <= {10 ** v}" for v in ([1, 2, 3] if "2" in subdata else [1, 2]) if "N" in columns] + ["T"], index=labels_list)
        print(str(df))
        latex += "\n\n" + subdata + "    (value, size, mse) at condition on N\n\n" + df.to_latex() + "\n"
    with open(file, "w") as f:
        f.write(latex + "\n\\end{document}")


if __name__ == "__main__":
    conditionN()
