"""
Stage 2: Autograd engine, built from scratch - no torch.autograd.

This file is a SCAFFOLD, not a solution. The Value class's public
contract (the methods below) is the whole thing. Fill in the bodies
yourself.

Recap of the mechanism (see Stage 1 theory + the worked traces for the
full derivation):
  - A Value wraps a number but also remembers what produced it (its
    children) and which operation created it, so the graph can be
    walked backward later. Plain eager arithmetic (just floats) throws
    this information away the moment it's computed - that's why Value
    needs to exist at all.
  - __add__ and __mul__ build the graph as a side effect of normal-
    looking arithmetic: each one computes the forward result AND
    records enough information (self, other, and which operation) for
    backward() to later know the LOCAL derivative rule to apply:
        d(a+b)/da = 1,       d(a+b)/db = 1
        d(a*b)/da = b.data,  d(a*b)/db = a.data
  - backward() has two jobs: (1) walk the graph in REVERSE topological
    order (never finalize a node's gradient until everything that
    consumed it has already contributed), and (2) at each node, use
    the chain rule to push gradient into its children - ACCUMULATING
    (+=) rather than overwriting, since a Value can be a child of more
    than one node (see the a*b + a*c worked trace - a's gradient is
    the SUM of both branches' contributions, not just one).
  - self.grad must start at 1 for whichever Value backward() is called
    on (dL/dL = 1 - a value's gradient with respect to itself), and 0
    for every other Value before any contributions arrive.

You will need to decide for yourself: how a Value remembers its
children, how it remembers which local-derivative rule to use when
backward() reaches it, and how you build the reverse-topological
order. All of that is your design - nothing here prescribes it.
"""

import math
class Value:
    def __init__(self, data,children=(),op=''):
        """Wrap a number as a graph node: grad starts at 0, and it remembers its children and the op that made it."""
        self.data = data
        self.grad = 0
        self.children=children
        self.op=op

    def __add__(self, other):
        """Return a new Value for self + other, recording both as children."""
        if not isinstance(other,Value):
                other=Value(other)
        return Value(self.data+other.data,children=(self,other),op='+')
    def __neg__(self):
        "Negates the Value, using -1, so as to support unary op -"
        return self*Value(-1)
    def __sub__(self,other):
        "subtracts two different Value objects and makes sure the other is also type casted to Value"
        if not isinstance(other,Value):
            other=Value(other)
        return self+(other)*-1
    def __radd__(self,other):
        return self+other
    def __rmul__(self,other):
        return self*other

    def __mul__(self, other):
        """Return a new Value for self * other, recording both as children."""
        if not isinstance(other,Value):
                other=Value(other)
        return Value(self.data * other.data, children=(self, other), op='*')

    def tanh(self):
        """Return a new Value for tanh(self), recording self as its child."""
        return Value(math.tanh(self.data),children=(self,),op='tanh')

    def exp(self):
        """Return a new Value for e ** self, recording self as its child."""
        return Value(math.exp(self.data),children=(self,),op='exp')

    def log(self):
        """Return a new Value for the natural log of self, recording self as its child."""
        return Value(math.log(self.data),children=(self,),op='log')

    def __pow__(self, k):
        """Return a new Value for self ** k (k is a plain number), remembering k for backward()."""
        out=Value(self.data**k, (self,),'pow')
        out.k=k
        return out 

    def build_topo_order(node, visited=None, order=None):
        if visited is None:
            visited = set()
        if order is None:
            order = []
        if node not in visited:
            visited.add(node)
            for child in node.children:
                Value.build_topo_order(child, visited, order)
            order.append(node)
        return order

    def backward(self):
        """Compute .grad for every Value that fed into this one, accumulating where a Value is used more than once."""
        topo = Value.build_topo_order(self)
        self.grad = 1

        for node in reversed(topo):
            if node.op == '+':
                node.children[0].grad += node.grad
                node.children[1].grad += node.grad
            elif node.op == '*':
                a, b = node.children
                a.grad += b.data * node.grad
                b.grad += a.data * node.grad
            elif node.op == 'tanh':
                child=node.children[0]
                child.grad+=(1-node.data**2)*node.grad
            elif node.op== 'exp':
                node.children[0].grad += node.data*node.grad
            elif node.op=='log':
                child=node.children[0]
                child.grad += (1/child.data)*node.grad if child.data !=0 else (1/child.data+0.0001)*node.grad
            elif node.op=='pow':
                child=node.children[0]
                child.grad += (node.k * (child.data)**(node.k-1))*node.grad
        
        
1