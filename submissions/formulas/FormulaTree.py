import numpy as np
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
        self.npar = 0
        self.add_node(0, "", 1)
        self.root = 0
        self.coefficients = np.array([])

    def input_data(self, X, y):
        self.X = X
        self.y = y

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

    def add_node(self, a, var="", nvar=5):
        if(var == ""):
            var_id = np.random.random_integers(0, nvar)
            var = "X[:," + str(var_id) + "]"
        i = len(self.nodes)
        if(self.verbose):
            print("adding node : ", i)
        self.nodes = np.append(self.nodes, Node(i, var, a, 0, 0))
        self.npar += 1
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
            self.npar -= 1
        return split

    def split_root(self, op="+", obj=""):
        n = len(self.nodes)
        self.add_node(n, op)
        self.add_node(n, obj)
        self.nodes[0].a = n
        self.nodes[n].b = 0
        self.nodes[n].c = n + 1
        self.root = n
        self.npar -= 1
        return self.root

    def add_coefficients(self, cname="C", constant=True):
        add_brackets = True
        c = 0
        for node in self.nodes:
            status = node.status
            if(status == 1):
                coefficient = cname + str(c)
                if(add_brackets):
                    coefficient = cname + "[" + str(c) + "]"
                self.split_node(node.id, "*", coefficient)
                self.npar -= 1
                c += 1
        if(constant):
            coefficient = cname + "[" + str(c) + "]"
            self.split_root("+", coefficient)

    def coefficient_error(self, c):
        C = c
        X = self.X
        y = self.y
        print(C)
        XF = eval(self.print_tree())
        e = y**2 - XF**2
        return np.sum(e)

    def fit_coefficients(self, X, y):
        self.input_data(X, y)
        c0 = np.full(self.npar, 0.5)
#        c0 = np.array(range(0, self.npar))
#        cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 2 * x[1] + 2},
#                {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
#                {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
        cons = ({})
        bnds = [[-5., 5.]] * self.npar
        print(bnds)
 
 #       res = minimize(fun=self.coefficient_error, x0=c0,
 #                      method='TNC', bounds=bnds,
 #                      options={'gtol': 0.0000001, 'eps': 0.000001, 'stepmx': 0.01})

 #       res = minimize(fun=self.coefficient_error, x0=c0,
 #                      method='dogleg', bounds=bnds,
 #                      options={'gtol': 0.01, 'initial_trust_radius': 0.001, 'max_trust_radius': 0.01})

        res = minimize(fun=self.coefficient_error, x0=c0,
                       method='COBYLA', bounds=bnds,
                       options={'gtol': 0.3, 'rhobeg': 0.1, 'maxiter': 1000000})

        print(res)
        if(res.success):
            self.coefficients = res.x
