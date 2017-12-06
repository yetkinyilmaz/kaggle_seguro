from submissions.formulas.GenerateTrees import *
from submissions.formulas.IO import *


def debug(n=100):

    trees = generatePopulation(2)
    write_trees(trees, 0, "debug_file.csv")
    trees = read_trees("debug_file.csv")
    trees = cross_population(trees)

debug()
