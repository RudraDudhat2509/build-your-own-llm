"""
Stage 4: scaled dot-product attention.

This file is a SCAFFOLD, not a solution. Fill in every function body
yourself. New functions get added to this file as the stage grows.

Pure Python only: vectors are plain lists of floats, matrices are lists
of lists, and the only import is `math`. No NumPy. This checks the
forward maths first; training through it (gradients) is a later
decision, not this one.

Every function below comes with a spec in the style of an interview
question: exact signature, exact output, and exact rules for bad input.
Every error in this file is a ValueError (with a short, clear message).

Recap of the mechanism (a small library: the QUERY is your question,
each KEY is a book label, each VALUE is that book's content). This
first batch covers the "which books match my question" half:
  - softmax: turns a list of numbers into probabilities (all positive,
    sum to 1). Bigger number -> bigger share.
        softmax([2, 1, 0]) = [0.665, 0.245, 0.090]
    It must be numerically stable: math.exp(1000) raises OverflowError
    in Python. Subtracting the maximum first fixes it and does NOT
    change the answer ([2,1,0] -> [0,-1,-2] gives the same shares).
    A score of -inf means "this position is blocked": it gets share 0.
    A real transformer uses this later to stop a word from looking at
    future words (the causal mask), so softmax must handle it now.
  - dot: multiply matching positions, then add them up.
        dot([1, 0], [1, 1]) = 1*1 + 0*1 = 1
  - attention_weights: for each query, score every key, then softmax.
        score   = dot(query, key) / sqrt(d)   one number per key
        weights = softmax(scores)             each row sums to 1
    d is the length of each query/key vector. Dividing by sqrt(d) stops
    the scores from growing as the vectors get longer.

Worked example (use it to check your own arithmetic):
    Q = [[1, 0]]
    K = [[1, 0], [0, 1], [1, 1]]
    scores         = [[1, 0, 1]]
    scores / 1.414 = [[0.707, 0, 0.707]]
    weights        = [[0.401, 0.198, 0.401]]
"""
import math


def softmax(x: list[float]) -> list[float]:
    """
    Return the softmax of `x` as a NEW list (same length as x).

        softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    Rules:
      - Raise ValueError if x is None or empty.
      - Raise ValueError if x contains nan or +inf.
      - -inf is allowed and gets probability exactly 0.0
        (softmax([0.0, -inf]) == [1.0, 0.0]).
      - Raise ValueError if EVERY value is -inf (there is nothing left
        to share the probability between).
      - Must be numerically stable: softmax([1000, 1001, 1002]) must
        not raise or give nan.
      - Must not change `x` itself.
    """
    if not x: 
        raise ValueError("For softmax, vector must not be None")
    negative_inf_count=0
    for val in x:
        if val==float('-inf'):
            negative_inf_count+=1
        if math.isnan(val): 
            raise ValueError("element is None")
        elif val==float('inf'):
            raise ValueError("element is Inf")
    if negative_inf_count==len(x):
        raise ValueError("all elements cannot be -inf")
    m=max(x)
    diff=[math.exp(xi-m) for xi in x ]
    total=sum(diff)
    if total:
        return [x/total for x in diff ]
    return [0]*len(x)

        



def dot(a: list[float], b: list[float]) -> float:
    """
    Return the dot product of a and b: a[0]*b[0] + a[1]*b[1] + ...

    Rules:
      - Raise ValueError if a or b is None or empty.
      - Raise ValueError if a and b have different lengths.
    """
    if not a or not b:
        raise ValueError("A or B is None")
    if len(a)!=len(b):
        raise ValueError("vectors dont have equal lengths")
    return sum(xi*yi for xi,yi in zip(a,b))

def attention_weights(Q: list[list[float]], K: list[list[float]]) -> list[list[float]]:
    """
    Q: n_queries vectors, each a list of d floats.
    K: n_keys vectors, each a list of d floats.
    Return n_queries rows, each a list of n_keys floats that sum to 1.

    For every query: score each key with dot(query, key) / sqrt(d),
    then softmax that row of scores. Use your own softmax() and dot().

    Rules:
      - Raise ValueError if Q or K is None or empty.
      - Raise ValueError if any vector in Q or K is empty.
      - Raise ValueError if the vectors do not all have the same
        length d (a ragged Q, a ragged K, or Q and K disagreeing).
      - Must not change Q or K.
    """
    if not Q or not K:
        raise ValueError("Q or K is None")
    ans=[]
    d=len(Q[0])
    for query in Q:
        if len(query)!=d:
            raise ValueError("Length unequal")
        if not query: 
            raise ValueError
        dots=[dot(query,key)/math.sqrt(d) for key in K]
        softmax_qk=softmax(dots)
        ans.append(softmax_qk)
    return ans
