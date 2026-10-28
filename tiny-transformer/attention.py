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

Third batch: the causal mask (no peeking at future words).
A language model reads a sentence and predicts the NEXT word. While
working on word i it may only look at words 0..i, never at later ones,
or it could just copy the answer. The mask is a grid of True/False:
position i may look at position j only when j <= i.
        looks at ->  w0  w1  w2
        word 0       yes no  no
        word 1       yes yes no
        word 2       yes yes yes
  - causal_mask: builds that grid for a sentence of n words.
  - attention_scores: the first half of attention_weights, stopped
    before softmax: just dot(query, key) / sqrt(d) for every pair.
  - apply_mask: replaces every blocked score with -inf. This must happen
    BEFORE softmax: softmax turns -inf into exactly 0 and shares the
    probability between the words that are left, so each row still sums
    to 1. (Zeroing the weights AFTER softmax is the classic mistake: the
    row no longer sums to 1.)
  - masked_attention: the whole path with a mask:
    scores -> apply_mask -> softmax each row -> blend V.

Worked example (use it to check your own arithmetic):
    Q = [[1, 0]]
    K = [[1, 0], [0, 1], [1, 1]]
    V = [[10, 0], [0, 10], [5, 5]]
    scores         = [[1, 0, 1]]
    scores / 1.414 = [[0.707, 0, 0.707]]
    weights        = [[0.401, 0.198, 0.401]]
    output         = [[6.017, 3.983]]

Masked worked example (3 words, each word is both query and key):
    Q = K = [[1, 0], [0, 1], [1, 1]]      V = [[10, 0], [0, 10], [5, 5]]
    scores / 1.414 = [[0.707, 0,     0.707],
                      [0,     0.707, 0.707],
                      [0.707, 0.707, 1.414]]
    after the mask = [[0.707, -inf,  -inf ],
                      [0,     0.707, -inf ],
                      [0.707, 0.707, 1.414]]
    weights        = [[1,     0,     0    ],      word 0 sees only itself
                      [0.330, 0.670, 0    ],
                      [0.248, 0.248, 0.503]]
    output         = [[10, 0], [3.302, 6.698], [5, 5]]
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


def causal_mask(n: int) -> list[list[bool]]:
    """
    Return an n-by-n grid of True/False: row i, column j is True when
    word i is allowed to look at word j, which means j <= i.

        causal_mask(3) == [[True,  False, False],
                           [True,  True,  False],
                           [True,  True,  True ]]

    Rules:
      - Raise ValueError if n is not an int. (True and False are not
        ints for this function, and neither are 2.0 or "3" or None.)
      - Raise ValueError if n < 1.
      - Every entry must be a real bool.
    """
    if not isinstance(n,int):
        raise ValueError("n should be int")
    if isinstance(n,int) and isinstance(n,bool):
        raise ValueError("N cannot be bool")
    if n<1:
        raise ValueError("n less than 1")
    return [[True if i>=j else False for j in range(n)] for i in range(n)]


def attention_scores(Q: list[list[float]], K: list[list[float]]) -> list[list[float]]:
    """
    Q: n_queries vectors of d floats. K: n_keys vectors of d floats.
    Return n_queries rows, each a list of n_keys floats:
        scores[i][j] = dot(Q[i], K[j]) / sqrt(d)

    This is attention_weights WITHOUT the softmax. The rows do not sum
    to 1 and can be negative.

    Rules (the same as attention_weights, for Q and K):
      - Raise ValueError if Q or K is None or empty.
      - Raise ValueError if any vector in Q or K is empty.
      - Raise ValueError if the vectors do not all have the same
        length d (a ragged Q, a ragged K, or Q and K disagreeing).
      - Must not change Q or K.
    """
    if not Q or not K:
        raise ValueError("No Q or K passed")
    d_q=len(Q[0])
    d_k=len(K[0])
    n_queries=len(Q)
    n_keys=len(K)
    if not d_q or not d_k:
        raise ValueError("vectors have no length")
    if d_q!=d_k:
        raise ValueError("Vectors in Q and K disagree on length")
    for q in Q:
        if q is None:
            raise ValueError("vector in Q is null")
        if len(q)!=d_q:
            raise ValueError("Vector in Q has diff length")
    for k in K:
        if k is None:
            raise ValueError("vector in K is null")
        if len(k)!=d_q:
            raise ValueError("Vector in K has diff length")
    ans=[]
    for i in range(n_queries):
        res=[]
        for j in range(n_keys):
            res.append(dot(Q[i],K[j])/math.sqrt(d_q))
        ans.append(res)
    return ans
    
        
    


def apply_mask(scores: list[list[float]], mask: list[list[bool]]) -> list[list[float]]:
    """
    Return a NEW grid the same shape as `scores`. Where mask is True the
    score is kept as it is. Where mask is False the score becomes
    float("-inf").

        apply_mask([[1.0, 2.0], [3.0, 4.0]], [[True, False], [True, True]])
        == [[1.0, -inf], [3.0, 4.0]]

    Rules:
      - Raise ValueError if scores or mask is None or empty, or if any
        row of either one is empty.
      - Raise ValueError if the two grids differ in shape (a different
        number of rows, or a row with a different length).
      - A row where EVERY entry is blocked is allowed here (softmax will
        be the one to reject it later).
      - Must not change scores or mask.
    """
    if not scores or not mask:
        raise ValueError("Scores and mask must not be None")
    r=len(scores)
    row=len(mask)
    c=len(scores[0])
    if r !=row:
        raise ValueError("incorrect shape")
    for i in range(row):
        if len(scores[i])!=len(mask[i]):
            raise ValueError("incorrect shape")
    return [[scores[i][j] if mask[i][j] else float('-inf') for j in range(c)] for i in range(r)]


def masked_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]],
                     mask: list[list[bool]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Attention where the mask decides who may look at whom.
    Q: n_queries vectors of d floats. K: n_keys vectors of d floats.
    V: n_keys vectors of d_v floats.
    mask: n_queries rows of n_keys bools (True = allowed to look).
    Return a tuple (output, weights):
      - output  has n_queries rows of d_v floats
      - weights has n_queries rows of n_keys floats; blocked positions
        are exactly 0.0 and every row still sums to 1

    Steps: attention_scores -> apply_mask -> softmax of each row ->
    apply_attention with V. Build it from your own functions.

    Rules:
      - Raise ValueError for every rule of attention_scores (Q and K),
        apply_mask (the mask), and apply_attention (V).
      - Raise ValueError if V has a different number of vectors than K.
      - Raise ValueError if the mask is not n_queries rows by n_keys
        columns.
      - Raise ValueError if some query is blocked from EVERY key (there
        is nothing to look at).
      - Must not change Q, K, V or mask.
    """
    if not Q or not K or not V:
        raise ValueError("Q K V must not be not null")
    n_keys=len(V)
    n_keys_k=len(K)
    if n_keys!=n_keys_k:
        raise ValueError("row count dont match in k and v")
    n_queries=len(Q)
    if not mask:
        raise ValueError("no mask given")
    if len(mask)!=n_queries or len(mask[0])!=n_keys:
        raise ValueError("shape mismatch for mask")
    scores=attention_scores(Q,K)
    masked_scores=apply_mask(scores,mask)
    softmaxed_scores=[softmax(x) for x in masked_scores]
    a_scores=apply_attention(softmaxed_scores,V)
    return (a_scores,softmaxed_scores)