
import numpy as np

from submissions.formulas.FormulaTree import *


def load_data(n_fit=1000000):
    train_filename = 'chopped_data/train.csv'
    data = pd.read_csv(train_filename).head(100)
    X_df = data.drop(['target'], axis=1)
    y = data[["target"]].values
    removed_column = 0
    X = np.delete(X_df.head(n_fit).values, removed_column, axis=1)
    y = y[0:n_fit]
    return X, y


def get_dataframe(tree):
    df = pd.DataFrame({'id': [], 'a': [], 'b': [], 'c': [],
                       'object': [], 'status': [], 'value': [], 'weight': []})
    ic = 0
    for node in tree.nodes:
        value = 0
        if("C" in node.object):
            value = tree.coefficients[ic]
            ic += 1
        df = df.append({'id': node.id, 'a': node.a, 'b': node.b, 'c': node.c,
                        'object': node.object, 'status': node.status,
                        'value': value, 'weight': 0}, ignore_index=True)
    return df


def write_tree(tree):
    df = tree.get_dataframe()
    df.to_csv("tree.csv")


def generateSingleTree(Nsplit=12):
    verbose = False
    ft = FormulaTree()
    n = 0

    while(n < Nsplit):
        maxnode = len(ft.nodes)
        split = False
        while(split is False):
            i = np.random.random_integers(0, maxnode - 1)
            if(verbose):
                print("request splitting node : ", i)
            split = ft.split_node(i)
        n += 1

    ft.add_coefficients()
    tree = ft.get_formula()
    print("Adding feature : ", tree)
    return ft


def generatePopulation(N=100):
    trees = np.array([])
    initial_depth = 3

    X, y = load_data()

    for i in range(0, N):
        tree = generateSingleTree(initial_depth)
        tree.set_classifier()
        tree.fit_coefficients(X, y)
        trees = np.append(trees, tree)

    for tree in trees:
        print("Tree Score : ", tree.score)

    return trees


def write_population(trees):
    data = pd.DataFrame()
    for i in range(0, len(trees)):
        tree = trees[i]
        n_nodes = len(tree.nodes)
        data_tree = pd.concat(
            [pd.DataFrame({'tree': [i] * n_nodes,
                           'score': [tree.score] * n_nodes}),
             get_dataframe(tree)],
            axis=1
        )
        data = data.append(data_tree, ignore_index=True)
    data.to_csv("population.csv")
    return data
