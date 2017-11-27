from submissions.formulas.GenerateTrees import *


def evolve(n=10):
    trees = generatePopulation()
    write_population(trees, 0, "output/generation_0.csv")
    for i in range(1, n):
        if(i % 2 == 0):
            trees = cross_population(trees)
        else:
            trees = propagate_population(trees)
        trees = cut_population(trees)
        file = "output/generation_" + str(i) + ".csv"
        write_population(trees, i, file)


evolve()
