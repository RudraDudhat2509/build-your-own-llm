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
    """Return the softmax of x as a new list. Stable for large values; -inf gets probability 0."""
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
    """Return the dot product of a and b."""
    if not a or not b:
        raise ValueError("A or B is None")
    if len(a)!=len(b):
        raise ValueError("vectors dont have equal lengths")
    return sum(xi*yi for xi,yi in zip(a,b))

def attention_weights(Q: list[list[float]], K: list[list[float]]) -> list[list[float]]:
    """For each query, score every key with dot / sqrt(d), then softmax each row."""
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
    """Blend the value vectors using each row of weights, giving one output row per query."""
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
    """Return (output, weights): attention_weights followed by apply_attention."""
    if not V or not K:
        raise ValueError()
    if len(V)!=len(K):
        raise ValueError('key and value length mismatch')
    w=attention_weights(Q,K)
    return (apply_attention(w,V),w)


def causal_mask(n: int) -> list[list[bool]]:
    """Return an n-by-n grid where position i may look at position j only when j <= i."""
    if not isinstance(n,int):
        raise ValueError("n should be int")
    if isinstance(n,int) and isinstance(n,bool):
        raise ValueError("N cannot be bool")
    if n<1:
        raise ValueError("n less than 1")
    return [[True if i>=j else False for j in range(n)] for i in range(n)]


def attention_scores(Q: list[list[float]], K: list[list[float]]) -> list[list[float]]:
    """Return dot(query, key) / sqrt(d) for every query-key pair, before softmax."""
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
    """Return a new grid with every blocked score (mask False) replaced by -inf."""
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
    """Return (output, weights) of attention where the mask decides who may look at whom."""
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