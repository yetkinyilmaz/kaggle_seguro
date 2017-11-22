
import numpy as np

from submissions.formulas.FormulaTree import *


def generateSingleTree(Nsplit=12):
    verbose = False
    ft = FormulaTree()
    n = 0

    operations = ["*", "+", "-"]

    while(n < Nsplit):
        maxnode = len(ft.nodes)
        split = False
        while(split == False):
            i = np.random.random_integers(0, maxnode - 1)
            if(verbose):
                print("request splitting node : ", i)
            split = ft.split_node(i)
        n += 1

    ft.add_coefficients()
    tree = ft.get_formula()
    print("Adding feature : ", tree)

    return ft


def generatePopulation(N=10):

    trees = np.array([])
    initial_depth = 3
    for i in range(0, N):
        trees = np.append(trees, generateSingleTree(initial_depth))

    return trees
