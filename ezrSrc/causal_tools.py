from math import *
from collections import Counter,defaultdict
from ezr import *

def disc(col, v, eps=1e-32):
  def z(x):   return (x - col.mu) / (col.sd + eps)
  def logistic(x):  return 1.0 / (1.0 + math.exp(-1.702 * z(x)))

  #   h = 3.5 * sd * len(v) ** (-1/3)
  #   print(h, int( (max(v)-min(v)) / h))
  #   if h < 0: q = 1
  #   max_bins, min_bins = 15, 3
  #   q = max(min_bins, min(int( (max(v)-min(v)) / h), int(max_bins)))
  if col.it == Sym: return v
  q=5
  # print("discr len =",len(v),"bins =", q)
  
  edges = [(i / q) for i in range(1, q)]
  ## Faster
  if v == "?": return
  p = logistic(v)
  return sum(1 for e in edges if p > e)

def disc2(col, v, eps=1e-32):
  def z(x):   return (x - col.mu) / (col.sd + eps)
  def logistic(x):  return 1.0 / (1.0 + math.exp(-1.702 * z(x)))

  #   h = 3.5 * sd * len(v) ** (-1/3)
  #   print(h, int( (max(v)-min(v)) / h))
  #   if h < 0: q = 1
  #   max_bins, min_bins = 15, 3
  #   q = max(min_bins, min(int( (max(v)-min(v)) / h), int(max_bins)))
  q=5
  # print("discr len =",len(v),"bins =", q)
  if col.it == Sym: return v
  edges = [(i / q) for i in range(1, q)]
  ## Faster
  bins = []
  for x in v:
    if x != "?":
      p = logistic(x)
      bins.append(sum(1 for e in edges if p > e))
  # print(bins)
  return bins
  ## Shorter
  # return [sum(1 for e in edges if logistic(x) > e) for x in v]

def mi(x,y,s=1e-32):
  n=len(x)
  nx,ny=Counter(x),Counter(y);nxy=defaultdict(int)
  for a,b in zip(x,y):nxy[(a,b)]+=1
  return sum((c/n)*log((c/n+s)/((nx[a]/n)*(ny[b]/n)+s),2)
             for (a,b),c in nxy.items())

def h(col, v):
  ## Entropy H(X) = -∑_{x∈X} p(x) log₂ p(x)
  x = disc2(col, v)
  n = len(x)
  counts = Counter(x)
  return -sum((c/n) * log(c/n, 2) for c in counts.values())

def hcond(xx, colx, yy, coly):
  ## Calculating the Conditional entropy H(X|Y)
  ## H(X|Y) = -∑∑ p(x,y) log₂ p(x|y)
  ## Conditional Probability Reminder: p(x|y) = p(x,y)/p(y)
  x = disc2(colx, xx)
  y = disc2(coly, yy)

  b=defaultdict(list);[b[bi].append(a) for a,bi in zip(x,y)]
  n=len(y)
  return sum(
    (len(v)/n) * (-sum((c/len(v))*log(c/len(v),2) 
    for c in Counter(v).values())) for v in b.values())

def micond(x,y,z,eps=1e-3):
  b=defaultdict(lambda:([],[]))
  for xi,yi,zi in zip(x,y,z):a,b_ = b[zi];a.append(xi);b_.append(yi)
  vals=[mi(a,b_) for a,b_ in b.values() if len(a)>1]
  vals, weights = [], []
  for a, b_ in b.values():
      if len(a) > 1:
          vals.append(mi(a, b_))
          weights.append(len(a))
  return (sum(v*w for v, w in zip(vals, weights)) / max(1, sum(weights)))

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

def causalTree(data, rows=None, Y=None, Klass=Num, how=None):
  "Prepare labeled data to pass it to Causal Tree Generator"
  Y = (lambda row: disty(data,row))
  def update_data(data):
    for col in data.cols.x:
      if col.it is Sym: 
        col.sd = div(col)
        col.mu = mid(col)
    ys = [Y(r) for r in data.rows]
    col = adds(ys)
    col.rows = ys
    data.ys = col
    return data

  def remove_confounder(data):
    ys = [Y(r) for r in data.rows]
    cf = []
    for col in data.cols.x:
      if mi(disc2(col, [r[col.at] for r in data.rows]) , disc2(data.ys, ys)) > 0.1:
        for c2 in data.cols.x:
          if c2 != col:
            if micond(disc2(col, [r[col.at] for r in data.rows]) , disc2(data.ys, ys), disc2(c2, [r[c2.at] for r in data.rows])) < 0.01:
              print(f"coofounder {col.txt} found!")
              cf.append(col)      
    
    names = [c.txt for c in cf]
    col_names = [c.txt if c.txt not in names else c.txt + "X" for c in data.cols.all ]
    new_data = Data([col_names] + data.rows)
    return update_data(new_data)

  the.leaf = 2
  return causalTreeGenerate(remove_confounder(update_data(data)))

def causalTreeSelects(col, row:Row, op:str, at:int, y:Atom) -> bool: 
  "Have we selected this row?"
  if (x:=row[at]) == "?" : return True
  if op == "<="          : return disc(col,x) <= y
  if op == "=="          : return disc(col,x) == y
  if op == ">"           : return disc(col,x) > y

def causalTreeGenerate(data, rows=None, Y=None, Klass=Num, how=None):
  "Create tree from list of lists using causality-based splits"
  DELTA = 0.02
  rows = rows or data.rows
  Y    = Y or (lambda row: disty(data,row))
  tree = o(rows=rows, how=how, kids=[], 
           mu=mid(adds(Y(r) for r in rows)))
  if len(rows) >= the.leaf:
    spread, cuts = min(causalCuts(data, c,rows,Y,Klass) for c in data.cols.x)
    if spread < 1 - DELTA:
      for cut in cuts:
        subset = [r for r in rows if causalTreeSelects(data.cols.all[cut[1]], r, *cut)]
        if the.leaf <= len(subset) < len(rows):
          tree.kids += [causalTreeGenerate(data, subset, Y, Klass, cut)]
  return tree

def causalCuts(data, col, rows, Y:callable, Klass:callable):
  "Return best cut for column at position 'at' using causality measure"
  xys = [(r[col.at], Y(r)) for r in rows if r[col.at] != "?"]
  return _causalCuts(data, col,xys,Y,Klass)

def _causalCuts(data, col,xys,Y,Klass) -> (float, list[Op]):
  "Causal cuts for symbolic column using H(Y|X)/H(Y) ratio."
  # Extract x and y values
  x_vals = [c[0] for c in xys]
  y_vals = [c[1] for c in xys]
    
  # Calculate H(Y) - entropy of target variable
  h_y = h(data.ys, y_vals)
  if h_y == 0 or len(x_vals) < 2:  
    return big, []

  # Calculate causality ratio: H(Y|X)/H(Y)
  # Lower ratio means X explains Y better (more causal)
  causality_ratio = (hcond(y_vals, data.ys, x_vals, col) + 1e-32) / h_y
  # print(f"hcond = {hcond(y_vals, x_vals):.2f}, ratio = {causality_ratio:.2f}")
  # Get unique symbolic values for cuts
  unique_x_vals = list(set(disc2(col,x_vals)))
  # print([("==", at, x) for x in unique_x_vals])

  return causality_ratio, [("==", col.at, x) for x in unique_x_vals]
