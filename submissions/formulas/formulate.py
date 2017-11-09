import numpy as np


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
        else:
            self.b = 0
            self.c = 0


class FormulaTree:
    def __init__(self):
        root = Node(0, "x", 1, 0, 0)
        self.nodes = np.array([root])

    def add_node(self, a):
        i = len(self.nodes)
        print("i : ", i)
        self.nodes = np.append(self.nodes, Node(i, "x", a, 0, 0))
        return i

    def split_node(self, inode):
        node = self.nodes[inode]
        status = node.status
        if(status != 1):
            print("cannot split intermediate node")
        else:
            node.status = 2
            node.object = "+"
            node.b = self.add_node(node.id)
            node.c = self.add_node(node.id)
            self.nodes[inode] = node


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


def encode(formula="formula"):

    #    def __init__(self, id, status, obj, a, b, c):

    ft = FormulaTree()
    tree = ft.nodes
    print(formula, " : ", print_tree(tree, 0))

    ft.split_node(0)
    tree = ft.nodes
    print(formula, " : ", print_tree(tree, 0))

