import pandas as pd
import numpy as np

from scipy.optimize import minimize

# Coefficient fitting related stuff.


def format_coefficients(tree, c):
    # add back the coeficients that were trivial in the fitting
    c = np.insert(c, 0, 1.)
    c = np.insert(c, tree.npar - 1, 0.)
    return c


def coefficient_target_error(tree, c):
    C = format_coefficients(tree, c)
    X = tree.X
    y = tree.y
    print(C)
    XF = eval(tree.get_formula())
    e = y**2 - XF**2
    return np.sum(e)


def coefficient_imbalance(tree, c):
    nc = len(c)
    C = format_coefficients(tree, c)
    X = tree.X
    print(c)
    DC = np.zeros(nc)
    XF = eval(self.get_formula())
#        print("XF : ", XF)
    e = 0
    norm = np.sum(XF**2)
    if(norm > 0):
        for i in range(0, nc):
            C = format_coefficients(tree, c)
            C[i] = 0
            XC0 = eval(tree.get_formula())
#                print("c : ", c)
#                print("C : ", C)
#                print("formula : ", self.print_tree())
#              print("XC0 : ", XC0)
            DC[i] = np.sum((XC0 - XF)**2) / norm
            e += DC[i]**2
            for j in range(0, i):
                e -= np.abs(DC[i] * DC[j])
    else:
        e = 1000000000000000.
    print("error : ", e)
    return np.sum(e)


def classifier_score(tree, c):
    C = format_coefficients(tree, c)
    X = tree.X
    y = tree.y

#    print(C)
    XF = eval(tree.get_formula()).reshape(-1, 1)
    X = np.delete(X, tree.variables, axis=1)
    X = np.concatenate((X, XF.reshape(-1, 1)), axis=1)
    score = tree.fit_classifier(X, y)

    print("classifier score : ", score)
    return score


def print_coefficients(tree):
    cs = "C = []"
    if(len(tree.coefficients) > 0):
        cs = "C = [" + tree.coefficients[0]
        for i in range(1, len(tree.coefficients)):
            cs = cs + "," + str(tree.coefficients[i])
        cs = cs + "]"
    print(cs)
    return cs


