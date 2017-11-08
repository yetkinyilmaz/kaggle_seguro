import numpy as np


class Node:
    def __init__(self, id, obj, a, b=0, c=0):
        # a -> b,  c
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
        else:
            self.b = 0
            self.c = 0


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
        text = ("(" +
                print_tree(tree, node.b) +
                node.object +
                print_tree(tree, node.c) +
                ")"
                )
    return text


def encode(formula="x + y"):

#    def __init__(self, id, status, obj, a, b, c):

    s = Node(0, "=", 0, 1, 1)
    n1 = Node(1, "*", 0, 2, 3)
    n2 = Node(2, "+", 1, 4, 5)

    n3 = Node(3, "x1", 2)
    n4 = Node(4, "x2", 1)
    n5 = Node(5, "x3", 2)

    tree = np.array([s, n1, n2, n3, n4, n5])

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
