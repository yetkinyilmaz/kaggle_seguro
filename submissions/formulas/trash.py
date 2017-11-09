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



