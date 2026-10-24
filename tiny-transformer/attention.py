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
each KEY is a book label, each VALUE is that book's content).
First batch: the "which books match my question" half:
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

Second batch: the "take what is inside the books" half:
  - apply_attention: each output row is a weighted blend of the value
    vectors, added up position by position. With weights [0.4, 0.2, 0.4]
    you take 40% of value 0, 20% of value 1 and 40% of value 2.
        0.4*[10, 0] + 0.2*[0, 10] + 0.4*[5, 5] = [4+0+2, 0+2+2] = [6, 4]
    The weights are used exactly as given: no normalizing, no checking
    that they sum to 1. (Two sanity checks: weights [0, 1, 0] return
    value 1 exactly; equal weights return the average of the values.)
  - attention: the whole path in one call. Weights from Q and K, then
    the blend of V. Returns both the output and the weights.

Worked example (use it to check your own arithmetic):
    Q = [[1, 0]]
    K = [[1, 0], [0, 1], [1, 1]]
    V = [[10, 0], [0, 10], [5, 5]]
    scores         = [[1, 0, 1]]
    scores / 1.414 = [[0.707, 0, 0.707]]
    weights        = [[0.401, 0.198, 0.401]]
    output         = [[6.017, 3.983]]
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


def apply_attention(weights: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    """
    weights: n_queries rows, each a list of n_keys floats.
    V: n_keys vectors, each a list of d_v floats.
    Return n_queries rows, each a list of d_v floats.

    Row i of the output is the blend of V's vectors using row i of
    weights, added up position by position:
        output[i][p] = weights[i][0]*V[0][p] + weights[i][1]*V[1][p] + ...

    Rules:
      - Raise ValueError if weights or V is None or empty.
      - Raise ValueError if any vector in V is empty, or if the vectors
        in V do not all have the same length d_v.
      - Raise ValueError if any row of weights does not have exactly
        len(V) numbers (one weight per value vector).
      - Use the weights exactly as given. Do NOT normalize them and do
        NOT check that they sum to 1 (weights [[2.0, 0.0, 0.0]] give
        2 times value 0).
      - Must not change weights or V.
    """
    if not weights or not V:
        raise ValueError("None value")
    for i in V:
        if len(i) != len(V[0]):
            raise ValueError("unequal row lengths")
    V_T=[list(row) for row in zip(*V)]
    d_v=len(V[0])
    n_keys=len(V)
    n_queries=len(weights)
    ans=[]
    if not d_v:
        raise ValueError("Empty list")
    for i in range(n_queries):
        layer_res=[]
        if len(weights[i])!=n_keys:
            raise ValueError("length mismatch")
        for p in range(d_v):
            layer_res.append(dot(weights[i],V_T[p]))
        ans.append(layer_res)
    return ans
            
        



def attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Q: n_queries vectors of d floats. K: n_keys vectors of d floats.
    V: n_keys vectors of d_v floats.
    Return a tuple (output, weights):
      - output  has n_queries rows of d_v floats
      - weights has n_queries rows of n_keys floats (rows sum to 1)

    Build it from your attention_weights() and apply_attention().

    Rules:
      - Raise ValueError for every rule of attention_weights (about Q
        and K) and every rule of apply_attention (about V).
      - Raise ValueError if V has a different number of vectors than K
        (there must be exactly one value per key).
      - Must not change Q, K or V.
    """
    if not V or not K:
        raise ValueError()
    if len(V)!=len(K):
        raise ValueError('key and value length mismatch')
    w=attention_weights(Q,K)
    return (apply_attention(w,V),w)