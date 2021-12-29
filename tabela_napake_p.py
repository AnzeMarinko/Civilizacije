from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from models import *
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = 'white'

printing_trees = False

data = pd.read_csv(f'{collected_folder}/meti_tabela_csv.csv', index_col=0)
parameters = list(data.columns)[1:-4]
parameters_nomath = [p.replace("_pm", "_{pm}").replace("_me", "_{me}") if "(" in p else p.replace("_", " ") for p in parameters]
label_columns = [("L", 'logL'), ("P(L < 1000)", 'L-1e3'), ("P(L < 10 000)", 'L-1e4'), ("P(L < 100 000)", 'L-1e5')]

dataTotal = np.load(f'{collected_folder}/meti_parameters.npy')
labelsTotal = np.load(f'{collected_folder}/meti_labels.npy')
columns = [[4, 6, 7, 9, 10, 11, 12], [4, 5, 8], [4, 6, 7, 9, 10, 11, 12], [6, 9, 13, 14, 15, 16, 17, 18]]

data = dataTotal[dataTotal[:, 0] == 1, 1:]
labels = labelsTotal[dataTotal[:, 0] == 1]
model1 = data[:, 0] == 1
model2 = data[:, 1] == 1
model3 = data[:, 2] == 1
model4 = data[:, 3] == 1
data2 = dataTotal[dataTotal[:, 0] == 2, 1:]
labels2 = labelsTotal[dataTotal[:, 0] == 2]
model3_2 = data2[:, 2] == 1
podatki = [[], [], [], [], [], [], []]
for stolpec in range(4):
    podatki[0] += [(data[model1, :][:, columns[0]], [parameters[c] for c in columns[0]],
                    labels[:, stolpec][model1], label_columns[stolpec], 'Model I')]
    podatki[1] += [(data[model2, :][:, columns[1]], [parameters[c] for c in columns[1]],
                    labels[:, stolpec][model2], label_columns[stolpec], 'Model II')]
    podatki[2] += [(data[model3, :][:, columns[2]], [parameters[c] for c in columns[2]],
                    labels[:, stolpec][model3], label_columns[stolpec], 'Model III')]
    podatki[3] += [(data2[model3_2, :][:, columns[2]], [parameters[c] for c in columns[2]],
                    labels2[:, stolpec][model3_2], label_columns[stolpec], 'Model III - 2')]
    podatki[4] += [(data[model4, :][:, columns[3]], [parameters[c] for c in columns[3]],
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
            return [[value[i]] + [n_node_samples[i] / n_node_samples[0] * 100, impurity[i]]]
        else:
            col = columns[feature[i]].replace("log(", "").replace(")", "")
            tresh = 10 ** round(threshold[i], 1) if "log" in columns[feature[i]] else round(threshold[i], 1)
            tresh = round(tresh * (100 if "f_" in col else 1), round(1 - threshold[i]) - (2 if "f_" in col else 0)) \
                if threshold[i] < 4 else f"{round(tresh / 1e10, 1)} * " + "10^{10}"
            tresh = f"{tresh}~\\%" if "f_" in col else tresh
            if "Model_Rare_Earth" in col:
                return [[("not Model IV", "Model IV", "Y", 0)] + r for r in recursive_rule(children_left[i])] + \
                       [[("Model IV", "Model IV", "N", 0)] + r for r in recursive_rule(children_right[i])]
            if "Model_Simplified" in col:
                return [[("not Model II", "Model II", "Y", 0)] + r for r in recursive_rule(children_left[i])] + \
                       [[("Model II", "Model II", "N", 0)] + r for r in recursive_rule(children_right[i])]
            return [[(f"{col} \\leq {tresh}", col, "<=", threshold[i])] + r for r in recursive_rule(children_left[i])] +\
                   [[(f"{col} > {tresh}", col, ">", threshold[i])] + r for r in recursive_rule(children_right[i])] + \
                   [[value[i]] + [n_node_samples[i] / n_node_samples[0] * 100, impurity[i]]]

    return [tuple(r) for r in recursive_rule()]


def modeli_napake(printing=False):
    size = 4
    plt.figure(figsize=(6 * size, 2 * size), dpi=300, tight_layout=True)
    pr = ""
    rules_exact0 = []
    for k, (X, columns, Y, label, subdata) in enumerate(podatki):
        # print(f"\r {round((j + 1)/20 * 100)}%", end=" ", flush=True)
        j = k // 4 + (k % 4) * 7
        plt.subplot(4, 7, j + 1)
        criterion = [("mse", "MSE"), ("mae", "MAE")][0]
        model = RandomForestRegressor(criterion=criterion[0], n_estimators=300, max_depth=4, random_state=1,
                                      min_samples_leaf=0.01)  # regresor
        model = model.fit(X, Y)
        feat_importances = pd.Series(model.feature_importances_,
                                     [f"${p}$" if "(" in p else p for p in columns])
        feat_importances.plot(kind='bar')
        if j // 7 == 0:
            plt.title(f"{subdata}")
        if j % 7 == 0:
            plt.ylabel(f"{label[0]}\nParameter importance")
        if j // 7 < 3:
            plt.xticks([])
        rules = []
        for i, tree in enumerate(model.estimators_):  # sestavi slovar in vrni nekaj pogostejših in njihovo pojavnost
            rules += rules_from_tree(tree, columns, printing_trees and i < 3)
        # print(len(rules), len(set(rules)))
        rules_raw = [tuple([r for r, _, _, _ in rule[:-3]] + [round(rule[-3], 1)]) for rule in rules]
        rules_text = [tuple([r for r, _, _, _ in rule[:-3]] + [round(rule[-3], 1)] + list(rule[-2:])) for rule in rules]
        rules_exact = [[(r1, r2, r3) for _, r1, r2, r3 in rule[:-3]] + [rule[-3:]] for rule in rules if rule[-1] < 0.2]
        # print(rules_exact)
        rules_raw = sorted([rule for rule in set(rules_raw)], key=lambda x: -rules_raw.count(x) / len(rules) * 100)
        ind_rules = np.array([[rules_raw.index(rule[:-2]), rule[-2], rule[-1]] for rule in rules_text])
        rules = [(rule, np.round(np.mean(ind_rules[ind_rules[:, 0] == i, 1]), 1),
                  np.round(np.mean(ind_rules[ind_rules[:, 0] == i, 2]), 3)) for i, rule in enumerate(rules_raw)]
        rules = [(rule, samples, gini_impurity) for rule, samples, gini_impurity in rules if gini_impurity < 0.2]
        # print(len(rules))
        land = "\\land"
        proc = '~\\%'
        if len(rules) > 0:
            pra = "\n\\subsection{" + f"{subdata}" + "}" + f"\n{label[0]}:\n" + "\\begin{itemize}\n"
            pra += "\n".join([f"\\item $({(') ' + land + ' (').join(r[0][:-1]) if len(r[0]) > 1 else 'T'}) \\Rightarrow "
                             f"{label[0]} = {r[0][-1] * 100 if 'P' in label[0] else 10 ** r[0][-1]:.1f}"
                             f"{proc if 'P' in label[0] else ''}$,\\hfill Size={round(r[1])} \\%, {criterion[1]}={r[2]}"
                              for r in rules[:25]])
            pra += "\n\\end{itemize}\n"
            if printing:
                print(pra, end="")
            pr += pra
        rules_exact0.append((subdata, label[0], rules_exact))
    if printing:
        with open("rules.txt", "w") as f:
            f.write(pr)
        plt.close()
    else:
        plt.savefig(f'slike/importance-random_forest_model.png')
        plt.show()
    with open("rules_exact.txt", "w") as f:
        f.write(str(rules_exact0))


if __name__ == "__main__":
    modeli_napake(True)
