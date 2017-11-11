import pandas as pd
import numpy as np

from scipy.optimize import minimize

from submissions.formulas.GenerateTrees import *


formula = generateSingleTree(2)
train_filename = 'kaggle_data/train.csv'
data = pd.read_csv(train_filename)
X = data.drop(['target'], axis=1).values
y = data[["target"]].values

def coefficient_score(x):

    return 0


x0 = np.array([1.3, 0.7])
res = minimize(rosen, x0,
               method='nelder-mead',
               options={'xtol': 1e-8, 'disp': True})
