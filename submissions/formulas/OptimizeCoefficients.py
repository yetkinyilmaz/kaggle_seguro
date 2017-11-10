import numpy as np

from scipy.optimize import minimize

from submissions.formulas.GenerateTrees import *


tree = generateSingleTree(2)
dataset

def coefficient_score(x):

    return 0


x0 = np.array([1.3, 0.7])
res = minimize(rosen, x0,
               method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})
