import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize


class Node:
    def __init__(self, id, obj, a, b=0, c=0):
        # a -> b,  c
        self.id = id
        self.a = a
        self.b = b
        self.c = c
        self.object = obj
        self.status = 1
        if((obj == "=") | (obj == "+") | (obj == "*")):
            self.status = 2
        # 0 : Nothing
        # 1 : End node
        # 2 : Branching operation
        # 3 : Coefficient
        # 4 : Solid

        else:
            self.b = 0
            self.c = 0


class FormulaTree:
    def __init__(self):
        self.verbose = False
        self.nodes = np.array([])
        self.nleaf = 0
        self.add_node(0, "", 1)
        self.root = 0
        self.coefficients = np.array([])

    def set_classifier(self):
        self.clf = DecisionTreeClassifier(
            max_depth=1,
            max_features=1,
            max_leaf_nodes=2
        )

    def input_data(self, X, y):
        self.X = X
        self.y = y

    def fit_classifier(self, X, y):
        self.clf.fit(X, y)
        return self.clf.score(X, y)

    def print_tree(self, inode=-1):
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
                    self.print_tree(node.b) +
                    node.object +
                    self.print_tree(node.c) +
                    ")"
                    )
        return text

    def add_node(self, a, var="", nvar=30):
        if(var == ""):
            var_id = np.random.random_integers(20, 20 + nvar)
            var = "X[:," + str(var_id) + "]"
        i = len(self.nodes)
        if(self.verbose):
            print("adding node : ", i)
        self.nodes = np.append(self.nodes, Node(i, var, a, 0, 0))
        self.nleaf += 1
        return i

    def split_node(self, inode, op="+", var="", var0="preserve"):
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
        self.add_node(n, op)
        self.add_node(n, obj)
        self.nodes[0].a = n
        self.nodes[n].b = 0
        self.nodes[n].c = n + 1
        self.root = n
        self.nleaf -= 1
        return self.root

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


    def format_coefficients(self, c):
        c = np.insert(c, 0, 1.)
        c = np.insert(c, self.npar-1, 0.)
        return c

    def coefficient_target_error(self, c):
        C = self.format_coefficients(c)
        X = self.X
        y = self.y
        print(C)
        XF = eval(self.print_tree())
        e = y**2 - XF**2
        return np.sum(e)

    def coefficient_imbalance(self, c):
        nc = len(c)
        C = self.format_coefficients(c)
        X = self.X
        print(c)
        DC = np.zeros(nc)
        XF = eval(self.print_tree())
#        print("XF : ", XF)
        e = 0
        norm = np.sum(XF**2)
        if(norm > 0):
            for i in range(0, nc):
                C = self.format_coefficients(c)
                C[i] = 0
                XC0 = eval(self.print_tree())
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

    def classifier_error(self, c):
        C = self.format_coefficients(c)
        X = self.X
        y = self.y

        print(C)
        XF = eval(self.print_tree()).reshape(-1, 1)
        score = self.fit_classifier(XF, y)
        error = 1. - score
        print("classifier error : ", error)
        return error


    def fit_coefficients(self, X, y):

        if(self.npar > 2):
            self.input_data(X, y)
            nc = self.npar-2
    #        c0 = np.full(nc, -100.002)
            c0 = 5. * (2. * np.random.random(nc) - 1.)
    #        c0 = np.array(range(0, self.npar))
    #        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
    #                {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
    #                {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
            cons = ({})
            bnds = [[-1000, 1000.]] * nc
            print(bnds)

    #        res = minimize(fun=self.classifier_error, x0=c0)


    #        res = minimize(fun=self.classifier_error, x0=c0,
    #                       method='L-BFGS-B', bounds=bnds,
    #                       options={'ftol': 0.000000001, 'gtol': 0.0000000001, 'eps': 1.003, 'maxcor': 1125, 'maxiter': 1000000})


            res = minimize(fun=self.classifier_error, x0=c0,
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
                self.coefficients = self.format_coefficients(res.x)
            else:
                self.coefficients = self.format_coefficients(c0)
        else:
            self.coefficients = np.array([1., 1.])

        print("Initial coefficients: ", self.format_coefficients(c0))
        print("Final   coefficients: ", self.coefficients)

