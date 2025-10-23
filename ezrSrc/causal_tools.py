from math import log
from collections import Counter,defaultdict
from ezr import *

def disc(v,q=16):
  idx=sorted(range(len(v)),key=lambda i:v[i]);r=[0]*len(v)
  for k,i in enumerate(idx):r[i]=k
  n=len(v)
  return [min(q-1,(k*q)//n) for k in r]

def disc2(v, mu, sd, q=10, eps=1e-32):
  def z(x):   return (x - mu) / (sd + eps)
  def logistic(x):  return 1.0 / (1.0 + math.exp(-1.702 * z(x)))
  
  edges = [(i / q) for i in range(1, q)]
  ## Faster
  bins = []
  for x in v:
      p = logistic(x)
      bins.append(sum(1 for e in edges if p > e))
  return bins
  ## Shorter
  # return [sum(1 for e in edges if logistic(x) > e) for x in v]

def mi(x,y,s=1e-32):
  n=len(x)
  nx,ny=Counter(x),Counter(y);nxy=defaultdict(int)
  for a,b in zip(x,y):nxy[(a,b)]+=1
  return sum((c/n)*log((c/n+s)/((nx[a]/n)*(ny[b]/n)+s),2)
             for (a,b),c in nxy.items())

def h(x):
    ## Entropy H(X) = -∑_{x∈X} p(x) log₂ p(x)
    n = len(x)
    counts = Counter(x)
    return -sum((c/n) * log(c/n, 2) for c in counts.values())

def hcond(x,y):
  ## Calculating the Conditional entropy H(X|Y)
  ## H(X|Y) = -∑∑ p(x,y) log₂ p(x|y)
  ## Conditional Probability Reminder: p(x|y) = p(x,y)/p(y)
  b=defaultdict(list);[b[bi].append(a) for a,bi in zip(x,y)]
  n=len(y)
  return sum((len(v)/n)*(-sum((c/len(v))*log(c/len(v),2)
          for c in Counter(v).values())) for v in b.values())

def micond(x,y,z,eps=1e-3,q=16):
  # if isinstance(x[0],(float,int)):x=disc(x,q)
  # if isinstance(y[0],(float,int)):y=disc(y,q)
  # if isinstance(z[0],(float,int)):z=disc(z,q)
  b=defaultdict(lambda:([],[]))
  for xi,yi,zi in zip(x,y,z):a,b_ = b[zi];a.append(xi);b_.append(yi)
  vals=[mi(a,b_) for a,b_ in b.values() if len(a)>1]
  return sum(vals)/max(1,len(vals))

def causal_ok(col,d2hs,rows,Zs=None,eps=1e-3,q=5,m=1e-6):
  ## col : column subject to check
  ## zs  : set of co-founder clomuns
  ## rows: all rows
  ## d2h : d2h of all rows
  x=[r[col.at] for r in rows]
  x_stats = adds(i for i in x)
  y=d2hs.rows
  if isinstance(x[0],(float,int)):x=disc2(x, x_stats.mu, x_stats.sd, q)
  if isinstance(y[0],(float,int)):y=disc2(y, d2hs.mu, d2hs.sd, q)
  print(f"Mutual-Info: {mi(x,y):.2f}, ", end ="")
  print(f"Conditionals: {hcond(x,y):.2f}, {hcond(y,x):.2f}, ", end ="\t")
  print(f"H(d2h|{col.txt}): {hcond(y,x):.2f}, {hcond(y,x):.2f}, ", end ="\t")
  if mi(x,y)<=eps:return False
  if hcond(x,y) > eps+hcond(y,x):return False
  if Zs:
    for Z in Zs:
      z = [r[Z.at] for r in rows]
      z_stats = adds(i for i in z)
      if isinstance(x[0],(float,int)):z=disc2(z, z_stats.mu, z_stats.sd, q)
      if micond(x,y,z,eps,q)<=eps:return False
  return True

# ## Tree Generation ----------------------------------------------------
def treeSelects(row:Row, op:str, at:int, y:Atom) -> bool: 
  "Have we selected this row?"
  if (x:=row[at]) == "?" : return True
  if op == "<="          : return x <= y
  if op == "=="          : return x == y
  if op == ">"           : return x > y

def CausalTree(data, rows=None, Y=None, Klass=Num, how=None):
  "Create tree from list of lists using causality-based splits"
  rows = rows or data.rows
  Y    = Y or (lambda row: disty(data,row))
  tree = o(rows=rows, how=how, kids=[], 
           mu=mid(adds(Y(r) for r in rows)))
  if len(rows) >= the.leaf:
    spread, cuts = min(CausalCuts(c,rows,Y,Klass) for c in data.cols.x)
    if spread < big:
      for cut in cuts:
        subset = [r for r in rows if treeSelects(r, *cut)]
        if the.leaf <= len(subset) < len(rows):
          tree.kids += [CausalTree(data, subset, Y, Klass, cut)]
  return tree

def CausalCuts(col, rows, Y:callable, Klass:callable):
  "Return best cut for column at position 'at' using causality measure"
  xys = sorted([(r[col.at], Y(r)) for r in rows if r[col.at] != "?"])
  return (_symCausalCuts if col.it is Sym else _numCausalCuts)(col.at,xys,Y,Klass)


def _symCausalCuts(at,xys,Y,Klass) -> (float, list[Op]):
  "Causal cuts for symbolic column using H(Y|X)/H(Y) ratio."
  # Extract x and y values
  x_vals = [c[0] for c in xys]
  y_vals = [c[1] for c in xys]
  
  # Calculate H(Y) - entropy of target variable
  h_y = h(y_vals)
  if h_y == 0:  return big, []
    
  # Calculate causality ratio: H(Y|X)/H(Y)
  # Lower ratio means X explains Y better (more causal)
  causality_ratio = (hcond(y_vals, x_vals) + 1e-32) / h_y
  # Get unique symbolic values for cuts
  unique_x_vals = list(set(x_vals))
  
  return causality_ratio, [("==", at, x) for x in unique_x_vals]

def _numCausalCuts(at,xys,Y,Klass) -> (float, list[Op]):
  "Causal cuts for numeric columns using H(Y|X)/H(Y) ratio with discretization."
  # Extract x and y values
  x_vals = [c[0] for c in xys]
  y_vals = [c[1] for c in xys]
  
  # Calculate H(Y) - entropy of target variable
  h_y = h(y_vals)
  if h_y == 0:  # If Y has no entropy, no point in splitting
    return big, []
  
  # Discretize x values for better causality detection
  x_stats = adds(x_vals)
  x_disc = disc2(x_vals, x_stats.mu, x_stats.sd, q=10)  # Use 10 bins for good granularity
  
  # Calculate causality ratio using discretized values
  h_y_given_x_disc = hcond(y_vals, x_disc)
  causality_ratio = h_y_given_x_disc / h_y
  
  # Find the best cut point based on discretized values
  best_cuts = []
  if causality_ratio < big:  # Only proceed if we have a valid ratio
    # Find the median of the discretized values to create a meaningful split
    unique_disc_vals = sorted(set(x_disc))
    if len(unique_disc_vals) > 1:
      # Use the middle value as the cut point
      cut_value = unique_disc_vals[len(unique_disc_vals)//2]
      # Convert back to original scale for the cut
      cut_x_val = x_stats.mu + (cut_value - 5) * x_stats.sd  # Approximate conversion
      best_cuts = [("<=", at, cut_x_val), (">", at, cut_x_val)]
  
  return causality_ratio, best_cuts
