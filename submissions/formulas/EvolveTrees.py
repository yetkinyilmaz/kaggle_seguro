from submissions.formulas.GenerateTrees import *
from submissions.formulas.IO import *


def evolve(n=10):
    for i in range(0, n):

        if(i == 0):
            trees = generatePopulation(20)
        else:
            trees = read_trees("output/generation_" + str(i - 1) + ".csv")
            if(i % 2 == 0):
                trees = cross_population(trees)
            else:
                trees = propagate_population(trees)

        trees = np.append(trees, generatePopulation(5))

        scores = np.array([])
        for tree in trees:
            scores = np.append(scores, tree.score)

        top_score = np.percentile(scores, 95)
        tolerance = 0.01
        print("Top 5% Score : ", top_score)
        trees = cut_population(trees, min_score=(top_score - tolerance))
        file = "output/generation_" + str(i) + ".csv"
        write_trees(trees, i, file)


evolve()
