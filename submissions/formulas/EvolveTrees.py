from submissions.formulas.GenerateTrees import *


def evolve(n=100):
    trees = generatePopulation(20)
    write_population(trees, 0, "output/generation_0.csv")
    for i in range(1, n):
        if(i % 2 == 0):
            trees = cross_population(trees)
        else:
            trees = propagate_population(trees)
        trees = np.append(trees, generatePopulation(5))
        trees = cut_population(trees, min_score=0.72)
        file = "output/generation_" + str(i) + ".csv"
        write_population(trees, i, file)


evolve()
