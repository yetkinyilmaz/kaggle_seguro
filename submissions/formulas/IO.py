import pandas as pd
import numpy as np
import re
from FormulaTree import *


def get_dataframe(tree):
    df = pd.DataFrame({'id': [], 'a': [], 'b': [], 'c': [],
                       'object': [], 'status': [], 'value': [], 'weight': []})
    ic = 0
    for node in tree.nodes:
        value = 0
        if("C" in node.object):
            value = node.value
            ic += 1
        df = df.append({'id': node.id, 'a': node.a, 'b': node.b, 'c': node.c,
                        'object': node.object, 'status': node.status,
                        'value': value, 'weight': 0}, ignore_index=True)
    return df


def write_tree(tree):
    df = tree.get_dataframe()
    df.to_csv("tree.csv")


def write_trees(trees, igen=0, file="population.csv"):
    data = pd.DataFrame()
    for i in range(0, len(trees)):
        tree = trees[i]
        n_nodes = len(tree.nodes)
        data_tree = pd.concat(
            [pd.DataFrame({'tree': [i] * n_nodes,
                           'generation': [igen] * n_nodes,
                           'score': [tree.score] * n_nodes}),
             get_dataframe(tree)],
            axis=1
        )
        data = data.append(data_tree, ignore_index=True)
    data.to_csv(file)
    return data


def read_trees(file):
    trees = np.array([])

    data = pd.read_csv(file)
    generation = data[["generation"]].values
    score = data[["score"]].values
    tree_ids = data[["tree"]].values

    for i in np.unique(tree_ids):
        print("reading tree : ", i)
        tree = FormulaTree()
        tree_nodes = tree_ids[:] == i
        tree.generation = generation[tree_nodes][0]
        tree.score = score[tree_nodes][0]

        node_id = data[["id"]].values[tree_nodes]
        node_a = data[["a"]].values[tree_nodes]
        node_b = data[["b"]].values[tree_nodes]
        node_c = data[["c"]].values[tree_nodes]
        node_object = data[["object"]].values[tree_nodes]
        node_status = data[["status"]].values[tree_nodes]
        node_value = data[["value"]].values[tree_nodes]
        node_weight = data[["weight"]].values[tree_nodes]

        for ino in node_id:
            inode = int(ino)
            print("inode : ", inode)
            node = Node(id=node_id[inode],
                        obj=node_object[inode],
                        value=node_value[inode],
                        weight=node_weight[inode],
                        a=node_a[inode],
                        b=node_b[inode],
                        c=node_c[inode]
                        )
            node.status = node_status[inode]
            if(node.a == -1):
                tree.root = node.id

            if(node.status == 1):
                # do coefficients and variables
                ivar = int(re.search(r'\d+', node.object).group())
                if("C" in node.object):
                    if not(str(ivar) in node.object):
                        print("Coefficients got reshuffled somewhere!!!!")
                    node.coefficients = np.append(tree.coefficients,
                                                  node.value)
                    print("Coefficient appended!!!")
                if("X" in node.object):
                    node.variables = np.append(tree.variables, ivar)

            tree.nodes = np.append(tree.nodes, node)

    trees = np.append(trees, tree)

    return trees
