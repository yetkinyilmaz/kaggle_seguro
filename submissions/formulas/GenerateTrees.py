
import numpy as np

from submissions.formulas.FormulaTree import *


def generateSingleTree(Nsplit = 12):

    ft = FormulaTree()
    n = 0

    operations = ["*", "+", "-"]

    while(n < Nsplit):
        maxnode = len(ft.nodes)
        split = False
        op = operations[np.random.random_integers(0, 2)]
        while(split == False):
            i = np.random.random_integers(0, maxnode - 1)
            print("request splitting node : ", i)
            split = ft.split_node(i, op)
        n += 1

    ft.add_coefficients()
    tree = ft.print_tree()
    print("Adding feature : ", tree)

    return tree


def generatePopulation(N=10):

    trees = np.array([])
    initial_depth = 3
    for i in range(0, N):
        trees = np.append(trees, generateSingleTree(initial_depth))

    return trees
