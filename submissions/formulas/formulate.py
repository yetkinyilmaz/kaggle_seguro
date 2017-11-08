import pandas as pd
import numpy as np


class Node:
    def __init__(self):
        # a -> b,  c
        self.a = 0
        self.b = 0
        self.c = 0
        self.object = Operation("+")
        self.status = 0
        # 0 : Nothing
        # 1 : End node
        # 2 : Branching operation


class Dimension:
    Spatial = -3.0
    Angular = 0.
    Order = 0
    Flat = True


class Operation:
    def __init__(self, sym="+"):
        self.symbol = sym
        self.a = "x"
        self.b = "x"
        self.c = ""

    def split_b(self, sym="+"):
        self.b = ""
        opb = Operation(sym)
        return opb

    d_a = Dimension()
    d_b = Dimension()
    d_c = Dimension()
    a = ""
    b = ""
    c = ""
    symbol = ""


class Generator:
    def __init__(self, dim=1, seed="+"):
        op = Operation(seed)
        self.ops = [op]

    def split(self, sign):
        self.ops = self.ops + [self.ops[len(self.ops) - 1].split_b(sign)]

    def formulate(self, complexity):
        formula = "p[0]+"
        ipar = 1
        for o in self.ops:
            for i in range(0, complexity):
                self.split(o)
            p = ""
            if o.symbol == "+":
                p = "p[" + str(ipar) + "]*"
                ipar += 1
            print(o.a, o.symbol, p, o.b)
            formula = formula + o.a + o.symbol + p + o.b
        return formula


def attach(text, obj):
    print("text: ", text, ",    obj: ", obj)
    return text + obj


def print_tree(tree, inode=0):
    print("printing tree starting from index ", inode)
    index = inode
    node = tree[index]
    status = node.status
    text = ""
    if(status == 1):
        text = node.object
        print(text)
    else:
        text = (print_tree(tree, node.b) +
                node.object +
                print_tree(tree, node.c))
    return text


def encode(formula="x + y"):

    s = Node()
    s.object = "="
    s.id = 0
    s.a = 0
    s.b = 1
    s.c = 1
    s.status = 0

    n0 = Node()
#    n0.object = Operation("+")
    n0.object = "+"
    n0.id = 1
    n0.a = 0
    n0.b = 2
    n0.c = 3
    n0.status = 2

    n1 = Node()
    n1.object = "x"
    n1.id = 2
    n1.a = 1
    n1.b = 0
    n1.c = 0
    n1.status = 1

    n2 = Node()
    n2.object = "+"
    n2.id = 3
    n2.a = 1
    n2.b = 4
    n2.c = 5
    n2.status = 2

    n3 = Node()
    n3.object = "x3"
    n3.id = 4
    n3.a = 3
    n3.b = 0
    n3.c = 0
    n3.status = 1

    n4 = Node()
    n4.object = "x4"
    n4.id = 5
    n4.a = 3
    n4.b = 0
    n4.c = 0
    n4.status = 1

    tree = np.array([s, n0, n1, n2, n3, n4])

    print(formula, " : ", print_tree(tree,0))


def funk(x, y, p=[], formula="p[0] + ( (p[1]*x) + ( p[2]*(x**2) ) )"):
    return eval(formula)


def test():
    #   return "2. * np.cos(np.arctan2(y,x)) / np.sqrt(x**2+y**2)"
    #  return "2. * y*y/x / np.sqrt(x**3+y**2+x**2)"
    myops = np.random.choice(["+", "*"], 20, p=[0.8, 0.2])
    gen = Generator("+")

    for o in myops:
        gen.split(o)

    formula = gen.formulate(12)

    return formula
