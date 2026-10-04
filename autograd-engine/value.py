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
        """
        Wrap a plain number in a Value. .grad starts at 0 - this
        Value hasn't received any gradient yet, and won't, unless
        backward() is eventually called on something downstream of it
        (or on this Value itself).
        """
        self.data = data
        self.grad = 0
        self.children=children
        self.op=op

    def __add__(self, other):
        """
        Return a NEW Value representing self + other. Must build the
        graph (remember self and other as this new Value's children)
        so backward() can later find its way back to them.
        """
        return Value(self.data+other.data,children=(self,other),op='+')

    def __mul__(self, other):
        """
        Return a NEW Value representing self * other. Same graph-
        building requirement as __add__.
        """
        return Value(self.data * other.data, children=(self, other), op='*')

    def tanh(self):
        """
        Return a NEW Value representing tanh(self.data). Unary op -
        one child, not two, so backward()'s dispatch needs a case
        that only unpacks one child instead of two.
        Local derivative: d(tanh(x))/dx = 1 - tanh(x)^2 - cheap to
        compute since the output Value's own .data already IS
        tanh(x), no need to recompute it.
        """
        return Value(math.tanh(self.data),children=(self,),op='tanh')

    def exp(self):
        """
        Return a NEW Value representing e^(self.data). Unary op.
        Local derivative: d(exp(x))/dx = exp(x) - the output is its
        own derivative, so again no extra computation needed beyond
        what's already stored in .data.
        """
        return Value(math.exp(self.data),children=(self,),op='exp')

    def log(self):
        """
        Return a NEW Value representing the natural log of
        self.data. Unary op.
        Local derivative: d(log(x))/dx = 1/x = 1/self.data.
        """
        return Value(math.log(self.data),children=(self,),op='log')

    def __pow__(self, k):
        """
        Return a NEW Value representing self.data ** k, where k is a
        plain number (int or float), NOT a Value - the exponent
        itself isn't a trainable parameter here, so it doesn't need
        its own graph node. Still unary (one child: self).
        Local derivative: d(x^k)/dx = k * x^(k-1).
        You'll need to remember k somewhere on the new Value so
        backward() can use it later when it reaches this node -
        where exactly is your call, same as everything else so far.
        This is what gives you division for free later: a * b**-1.
        """
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
        """
        Compute the gradient of every Value that fed into this one
        (directly or indirectly), and store it in each Value's .grad.
        Seeds self.grad = 1 first (this Value's gradient with respect
        to itself). Must accumulate contributions (+=), never
        overwrite, for any Value used in more than one place.
        """
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