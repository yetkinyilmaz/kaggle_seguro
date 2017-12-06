from submissions.formulas.GenerateTrees import *
from submissions.formulas.IO import *
from submissions.formulas.OptimizeCoefficients import *


trees = read_trees("output/generation_5.csv")
for tree in trees:
    tree.verbose = False
#   tree.print_nodes()
    print("Formula : ", tree.get_formula(), ", Score : ", tree.score)
    print(print_coefficients(tree))
