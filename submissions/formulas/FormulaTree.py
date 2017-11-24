import numpy as np

from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize
from submissions.formulas.OptimizeCoefficients import *


class Node:
    def __init__(self, id, obj, a, b=-1, c=-1):
        # a -> b,  c
        self.id = id
        self.a = a
        self.b = b
        self.c = c
        self.object = obj
        self.value = 1.
        self.weight = 1.
        self.status = 1
        if((obj == "=") | (obj == "+") | (obj == "*")):
            self.status = 2
        # -1 : Dead
        #  0 : Nothing
        #  1 : End node
        #  2 : Branching operation
        #  3 : Coefficient
        #  4 : Solid
        else:
            self.b = 0
            self.c = 0


class FormulaTree:
    def __init__(self):
        self.verbose = False
        self.nleaf = 0
        self.root = 0
        self.score = 0
        self.coefficients = np.array([])
        self.variables = np.array(range(20, 30))
        self.operations = ["*", "+", "-"]
        self.nodes = np.array([])
        self.add_node(0, "")

    def set_classifier(self):
        self.clf = DecisionTreeClassifier(
            max_depth=1,
            max_features=1,
            max_leaf_nodes=2
        )

    def error_function(self, c):
        return 1. - classifier_score(self, c)

    def input_data(self, X, y):
        self.X = X
        self.y = y

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)
        return self.clf.score(X, y)

    def get_subtree(self, inode):
        subtree = FormulaTree()
        pass
        return subtree

    def get_formula(self, inode=-1):
        if(inode < 0):
            inode = self.root
        if(self.verbose):
            print("printing tree starting from index ", inode)
        index = inode
        node = self.nodes[index]
        status = node.status
        text = ""
        if(status == 1):
            text = node.object
            if(self.verbose):
                print(text)
        else:
            text = ("(" +
                    self.get_formula(node.b) +
                    node.object +
                    self.get_formula(node.c) +
                    ")"
                    )
        return text

    def add_node(self, a=-1, var=""):
        if(var == ""):
            var_id = self.variables[
                np.random.random_integers(0, len(self.variables) - 1)
            ]
            var = "X[:," + str(var_id) + "]"
        i = len(self.nodes)
        if(self.verbose):
            print("adding node : ", i)
        self.nodes = np.append(self.nodes, Node(i, var, a, -1, -1))
        self.nleaf += 1
        return i

    def kill_node(self, inode):
        if(self.verbose):
            print("Killing node : ", inode)
        status = self.nodes[inode].status
        self.nodes[inode].status = -1
        if(status == 1):
            self.nleaf -= 1
        if(status == 2):
            self.kill_node(self.nodes[inode].b)
            self.kill_node(self.nodes[inode].c)

    def clean_dead(self):
        for i in range(len(self.nodes) - 1, -1, -1):
            if(self.nodes[i].status == -1):
                self.nodes = np.delete(self.nodes, i)

    def reset_ids(self, offset=0):
        new_ids = np.array(range(0, len(self.nodes)))
        ids = np.array([])
        for node in self.nodes:
            if(node.status != -1):
                ids = np.append(ids, int(node.id))

        if(self.verbose):
            print("ids     (", len(ids), ") : ", ids)
            print("new_ids (", len(new_ids), ") : ", new_ids)

        for i in range(0, len(self.nodes)):
            if(self.nodes[i].status != -1):

                if(self.verbose):
                    print("node i : ", i)
                    print("a : ", self.nodes[i].a)
                    print("b : ", self.nodes[i].b)
                    print("c : ", self.nodes[i].c)

                new_id, a, b, c = -1, -1, -1, -1
                new_id = np.where(ids == self.nodes[i].id)[0][0] + offset
                if(self.nodes[i].id != self.root):
                    a = np.where(ids == self.nodes[i].a)[0][0] + offset
                else:
                    self.root = new_id
                if(self.nodes[i].status == 2):
                    b = np.where(ids == self.nodes[i].b)[0][0] + offset
                    c = np.where(ids == self.nodes[i].c)[0][0] + offset

                self.nodes[i].a = a
                self.nodes[i].b = b
                self.nodes[i].c = c
                self.nodes[i].id = new_id

    def add_coefficients(self, cname="C", constant=True):
        add_brackets = True
        c = 0
        self.npar = 0
        for node in self.nodes:
            status = node.status
            parent = self.nodes[node.a]
            if(status == 1 and
                    not ((parent.object == "*") & (parent.c == node.id))):
                coefficient = cname + str(c)
                if(add_brackets):
                    coefficient = cname + "[" + str(c) + "]"
                self.split_node(node.id, "*", coefficient)
                self.nleaf -= 1
                self.npar += 1
                c += 1
        if(constant):
            coefficient = cname + "[" + str(c) + "]"
            self.split_root("+", coefficient)
            self.npar += 1
        self.coefficients = np.full(self.npar, 1.)

    def fit_coefficients(self, X, y):
        self.score = 0
        c0 = 5. * (2. * np.random.random(self.npar) - 1.)

        if(self.npar > 2):
            self.input_data(X, y)
            nc = self.npar - 2
    #        c0 = np.full(nc, -100.002)
            c0 = 5. * (2. * np.random.random(nc) - 1.)

            bnds = [[-1000, 1000.]] * nc
            print(bnds)

# Fitting coefficients
    #        res = minimize(fun=self.classifier_error, x0=c0)

    #        res = minimize(fun=self.classifier_error, x0=c0,
    #                       method='L-BFGS-B', bounds=bnds,
    #                       options={'ftol': 0.000000001, 'gtol': 0.0000000001, 'eps': 1.003, 'maxcor': 1125, 'maxiter': 1000000})

            res = minimize(fun=self.error_function, x0=c0,
                           method='L-BFGS-B',
                           #                           bounds=bnds,
                           options={'xtol': 0.001, 'eps': 0.2, 'maxiter': 1000000})

    #        res = minimize(fun=self.classifier_error, x0=c0,
    #                       method='SLSQP', bounds=bnds,
    #                       options={'ftol': 0.00001, 'eps': 1110.05, 'maxiter': 1000000})

#            res = minimize(fun=self.classifier_error, x0=c0,
#                           method='BFGS',
#                           bounds=bnds,
#                           options={'gtol': 0.9, 'eps': 0.5, 'norm': 10.1, 'maxiter': 1000000})

            print(res)
            if(res.success):
                self.coefficients = format_coefficients(self, res.x)
                self.score = 1. - res.fun
            else:
                self.coefficients = format_coefficients(self, c0)
        else:
            self.coefficients = format_coefficients(self, c0)

        print("Initial coefficients: ", format_coefficients(self, c0))
        print("Final   coefficients: ", self.coefficients)
        return self.score

# external modification methods

    def split_node(self, inode, op="", var="", var0="preserve"):
        if(op == ""):
            op = self.operations[
                np.random.random_integers(0, len(self.operations) - 1)
            ]

        if(self.verbose):
            print("splitting node : ", inode)
        node = self.nodes[inode]
        status = node.status
        split = False
        if(status != 1):
            if(self.verbose):
                print("cannot split intermediate node or coefficient")
            split = False
        else:
            if(var0 == "preserve"):
                var0 = node.object
            node.b = self.add_node(node.id, var)
            node.c = self.add_node(node.id, var0)
            node.status = 2
            node.object = op
            self.nodes[inode] = node
            split = True
            self.nleaf -= 1
        return split

    def split_root(self, op="+", obj=""):
        n = len(self.nodes)
        self.add_node(-1, op)
        self.add_node(n, obj)
        self.nodes[self.root].a = n
        self.nodes[n].b = self.root
        self.nodes[n].c = n + 1
        self.root = n
        self.nleaf -= 1
        return self.root

    def replace_node(self, inode, obj=""):
        node = self.nodes[inode]
        if(obj == ""):
            if(node.status == 1):
                var_id = self.variables[
                    np.random.random_integers(0, len(self.variables) - 1)
                ]
                var = "X[:," + str(var_id) + "]"
                obj = var
            else:
                op = self.operations[
                    np.random.random_integers(0, len(self.operations) - 1)
                ]
                obj = op
        node.object = obj
        self.nodes[inode] = node

    def truncate_node(self, inode, obj="END"):
        if(inode == self.root):
            "Truncating the whole tree"
        if(self.verbose):
            print("truncating tree starting from index ", inode)
        self.kill_node(inode)
        self.nleaf += 1
        self.nodes[inode].object = obj
        self.nodes[inode].status = 1
        self.nodes[inode].b = -1
        self.nodes[inode].c = -1
        self.reset_ids()
        self.clean_dead()
        return inode

    def reconnect_node(self, inode, subtree):
        node = self.nodes[inode]
        pass

