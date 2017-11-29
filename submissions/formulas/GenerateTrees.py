
import numpy as np
import copy

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


def generateSingleTree(Nsplit=1):
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
    X, y = load_data()

    for i in range(0, N):
        tree = generateSingleTree()
        tree.set_classifier()
        tree.fit_coefficients(X, y)
        trees = np.append(trees, tree)

    for tree in trees:
        print("Tree Score : ", tree.score)

    return trees


def propagate_population(trees):
    children = trees
    for tree in children:
        tree.add_subtrees(trees)
        tree = mutate_tree(tree)
    return children


def cross_population(trees):
    n = len(trees)
    children = copy.deepcopy(trees)
    for i in range(0, n):
        trees[i].add_subtrees(trees)
        ic = -1
        while(ic < 0 | ic == i):
            ic = np.random.random_integers(0, n - 1)
        children[i] = cross_trees(trees[i], trees[ic])
    return children


def cut_population(trees, min_score=0.6):
    survivors = np.array([])
    for tree in trees:
        if(tree.score > min_score):
            survivors = np.append(survivors, tree)
    return survivors


def mutate_tree(tree):
    split = False
    while(split is False):
        inode = np.random.random_integers(0, len(tree.nodes) - 1)
        split = tree.split_node(inode)
    return tree


def cross_trees(tree, subtree):
    crossed = False
    while(crossed is False):
        inode = np.random.random_integers(0, len(tree.nodes) - 1)
        crossed = tree.graft_node(subtree, inode)
    return tree



