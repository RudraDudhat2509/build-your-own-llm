"""
Stage 3: basic neural network / char-level language model.

This file is a SCAFFOLD, not a solution. Fill in every function/method
body yourself. New functions get added to this file as the stage grows.

Depends on the Value class from the autograd-engine repo (sibling
folder). Imported via a relative path for now - no packaging yet.

Every function below comes with a spec in the style of an interview
question: exact signature, exact output, and exact rules for bad input.
Every error in this file is a ValueError (with a short, clear message).

Batch 1 (vocabulary + loss): two small, independent functions.

  - build_vocab: a character vocabulary is just every distinct
    character in the training text, each given an integer id.
        build_vocab("cab") -> stoi={'a':0,'b':1,'c':2}, itos={0:'a',1:'b',2:'c'}
    (sorted order, so it's deterministic - same text always gives the
    same ids).

  - softmax_cross_entropy: the loss for one prediction. Turn the
    logits into probabilities (softmax), then take -log of the
    probability assigned to the correct target.
        logits = [011.0, 1.0, -1.0], target_id = 1
        exp:   [1.0, 2.718, 0.368]
        sum  = 4.086
        probs = [0.245, 0.665, 0.090]
        loss = -log(probs[1]) = -log(0.665) = 0.408
    Built entirely from Value's exp/pow/log - no new autograd machinery.

Batch 2 (the model): CharMLP's four methods, not started yet - see
their docstrings below for what each one needs to do.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "autograd-engine"))
from value import Value  # noqa: E402


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Return (stoi, itos): stoi maps each distinct character in `text`
    to a unique integer id, itos is the reverse mapping.

    Rules:
      - Raise ValueError if text is None or empty.
      - Ordering must be deterministic (sorted() is the natural choice).
    """
    if not text:
        raise ValueError("Text provided is none")
    stoi={}
    for i in sorted(text):
        if i in stoi:
          continue
        stoi[i]=len(stoi)
    itos={v:k for k,v in stoi.items()}
    return (stoi,itos)
    
    

import math
def softmax_cross_entropy(logits: list[Value], target_id: int) -> Value:
    """
    logits: list of Value objects (one per vocabulary entry, NOT yet
    softmax-normalized).
    target_id: the integer id of the correct next character.
    Returns: a single Value - the loss - built from exp, the sum of
    exps, pow(-1) for the division, and log, exactly like the worked
    example above.

    Rules:
      - Raise ValueError if logits is None or empty.
      - Raise ValueError if target_id is not a valid index into logits
        (not an int, is a bool, or out of range 0 <= target_id < len(logits)).

    A note on sum(): Python's built-in sum() starts its accumulator at
    the plain int 0, and Value only defines __add__, not __radd__ - so
    sum([v1, v2, v3]) tries `0 + v1` first and fails. Either pass an
    explicit Value(0.0) start, or accumulate with a plain loop.
    """
    if not logits:
        raise ValueError("Logits empty")
    if target_id not in range(len(logits)):
        raise ValueError("index not in logits")
    if not isinstance(target_id,int):
        raise ValueError("not a int")
    if isinstance(target_id,bool):
        raise ValueError("not a int")
    exp_list=[x.exp() for x in logits]
    sum=Value(0.0)
    for i in exp_list:
        sum+=i
    exp_list = [e * (sum** -1) for e in exp_list]
    return exp_list[target_id].log()*Value(-1)




class CharMLP:
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, context_size: int):
        """
        Build every trainable Value this model needs:
          - an embedding table: vocab_size rows, embedding_dim Values each
          - a hidden layer: hidden_dim rows of (embedding_dim * context_size)
            weights each, plus hidden_dim biases
          - an output layer: vocab_size rows of hidden_dim weights each,
            plus vocab_size biases
        Random-initialize weights (random.uniform(-1, 1) is fine);
        biases can start at 0.

        Rules:
          - Raise ValueError if any of vocab_size, embedding_dim,
            hidden_dim, context_size is not a positive int.
        """
        if vocab_size<0 or embedding_dim<0 or hidden_dim<0 or context_size<0:
          raise ValueError("Not positive")
        self.embedding = [[Value(random.uniform(-1, 1)) for _ in range(embedding_dim)]
    for _ in range(vocab_size)]
        self.W1=[[Value(random.uniform(-1,1)) for _ in range(embedding_dim*context_size)] for _ in range(hidden_dim)]
        self.W2=[[Value(random.uniform(-1,1)) for _ in range(hidden_dim)] for _ in range(vocab_size)]
        self.b1=[Value(0.0) for _ in range(hidden_dim)]
        self.b2=[Value(0.0) for _ in range(vocab_size)]



    def parameters(self) -> list[Value]:
        """
        Return a flat list of every trainable Value in this model -
        every embedding entry, every weight, every bias.
        """
        embed_list=[x for row in self.embedding for x in row]
        W1_list=[x for row in self.W1 for x in row]
        W2_list=[x for row in self.W2 for x in row]
        return embed_list+W1_list+W2_list+self.b1+self.b2

    def zero_grad(self) -> None:
        """Reset every parameter's .grad to 0."""
        for x in self.parameters():
            x.grad=0
            
    

    def forward(self, context_ids: list[int]) -> list[Value]:
        """
        context_ids: a list of character ids, length == self.context_size.
        Returns: a list of vocab_size Value objects (the logits - NOT
        yet softmax-normalized; that happens in softmax_cross_entropy).

        Rules:
          - Raise ValueError if context_ids is None, or its length
            does not equal self.context_size.
          - Raise ValueError if any id in context_ids is not a valid
            index into the embedding table (0 <= id < vocab_size).
        """
        h=[]
        x = []
        for cid in context_ids:
            x.extend(self.embedding[cid])
        for row,bias in zip(self.W1,self.b1):
            s=sum(w*x1 for w,x1 in zip(row,x))+bias
            h.append(s.tanh())
        logits=[]
        for row,bias in zip(self.W2, self.b2):
            s=sum(w*hi for w,hi in zip(row,h))+bias
            logits.append(s)
        return logits
        
