from fileinput import filename
from ezr import *
import matplotlib.pyplot as plt
import numpy as np
from causal_tools import *
import stats as stats
import stats2 as stats2


def apply_ezr(data):
  b4   = adds(disty(data,row) for row in data.rows)
  data.rows = shuffle(data.rows)
  win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
  best = lambda rows: win(disty(data, distysort(data,rows)[0]))
  
  seed = 1
  stp = 50
  the.Budget = stp
  the.acq = "near"
  the.few = 256
  the.Any = 5
  picks = 10
  the.seed = seed
  random.seed = seed

  labels = likely(data)
  tree = Tree(clone(data, labels))
  # row_mu_sorted = sorted([(row, treeLeaf(tree, row).mu) for row in data.rows], key=lambda x: x[1])
  # ii = 0
  # while row_mu_sorted[ii][1] == row_mu_sorted[ii+1][1]:
  #     if ii<len(row_mu_sorted)-1: ii += 1
  #     else: break
  # # print("How many falls under best branch?", ii,"out of", len(row_mu_sorted))
  # selection = random.sample(row_mu_sorted[:ii], min(ii,picks))
  # best_win_rate = best([i[0] for i in selection])

def visualize(data, labels):
  d2hs   = adds(disty(data,row) for row in labels)
  d2hs.rows = [disty(data,row) for row in labels]
  for col in data.cols.x:
    print(f"working on {col.txt}")
    res = causal_ok(col, d2hs, labels, [c for c in data.cols.x if c != col])
    print("Causal :",res)
    x=[r[col.at] for r in labels]
    if isinstance(x[0],(float,int)):x=disc(x,5)
    if isinstance(d2hs[0],(float,int)):d2hs=disc(d2hs,5)
    col_values = [row[col.at] for row in labels]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, d2hs, alpha=0.6, s=20)
    plt.xlabel(f'{col.txt}')
    plt.ylabel('D2h')
    plt.grid(True, alpha=0.3)
    
    if len(col_values) > 1 and len(d2hs) > 1:
        corr_coef = np.corrcoef(col_values, d2hs)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}, Corr: {res}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.show()

def play():
  # filename = sys.argv[1]
  filename = "../Data/moot/auto93.csv"
  data = Data(csv(filename))
  half = len(data.rows) // 2
  data.rows = shuffle(data.rows)
  train, holdout = data.rows[:half], data.rows[half:]

  b4   = adds(disty(data,row) for row in data.rows)
  win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
  d2hs = [disty(data,row) for row in data.rows]
  stp = 50
  the.Budget = stp
  labels = likely(clone(data, train))
  tree = Tree(clone(data, labels))
  treeShow(data, tree, win)
  for level, node in treeNodes(tree):
    node_rows = node.rows
    if level == 0:
      print(f"Level 0 with {len(node_rows)} rows:")
      for col in data.cols.x:
        d2hs   = adds(disty(data,row) for row in node_rows)
        d2hs.rows = [disty(data,row) for row in node_rows] 
        print(f"{col.txt}\t--> mu:{node.mu:.2f},\t", end = "")
        res = causal_ok(col, d2hs, node_rows, [c for c in data.cols.x if c != col])
        print(f"Causal:{res}")
    print()
    for kid in node.kids:
      print(f"  Level {level+1} with {len(kid.rows)} rows:")
      op, at, y = kid.how
      chosen_col = data.cols.x[at]      
      rule = f"if {data.cols.names[at]} {op} {y}"
      print(f"  Rule:\t{rule if hasattr(node, 'how') else 'Root'}")
      print("  Before:\t\t", end = "")
      d2hs   = adds(disty(data,row ) for row in node_rows)
      d2hs.rows = [disty(data,row) for row in node_rows] 
      print(f"  {chosen_col.txt}\t--> mu:{node.mu:.2f},\t", end = "")
      res = causal_ok(chosen_col, d2hs, node_rows, [c for c in data.cols.x if c != chosen_col])
      print(f"Causal:{res}")

      print("  After:\t\t", end = "")
      d2hs   = adds(disty(data,row ) for row in kid.rows)
      d2hs.rows = [disty(data,row) for row in kid.rows] 
      print(f"  {chosen_col.txt}\t--> mu:{kid.mu:.2f},\t", end = "")
      res = causal_ok(chosen_col, d2hs, kid.rows, [c for c in data.cols.x if c != chosen_col])
      print(f"Causal:{res}")
      print()
    print("---------")
    input()
    
    # visualize(data, node.rows)

  # [print(names) for names in data.cols.x]


def compareTrees(data):
  half = len(data.rows) // 2
  data.rows = shuffle(data.rows)
  train, holdout = data.rows[:half], data.rows[half:]
  b4   = adds(disty(data,row) for row in data.rows)
  win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
  best = lambda rows: win(disty(data, distysort(data,rows)[0]))
  the.Budget = 50
  the.Check = 10
  labels = likely(clone(data, train))
  
  asIs, ezr, causal = 1e-32, 1e-32, 1e-32
  ## Treatments
  for trt in ["causal", "ezr", "asIs"]:
    if trt == "ezr":      tree   = Tree(clone(data, labels))
    elif trt == "causal": tree   = CausalTree(clone(data, labels))
    else:  
      asIs = win(disty(data, random.sample(holdout, k=1)[0]))
      continue


    print(f"{trt} Tree :")
    treeShow(data, tree,win)
    best_holdout = best(sorted(holdout, key=lambda row: treeLeaf(tree,row).mu)[:the.Check])
    if trt == "ezr":      ezr    = best_holdout
    elif trt == "causal": causal = best_holdout

    print(f"Best train: {best(labels)}, Best hold-out: {best_holdout}")

    row_mu_sorted = sorted([(row, treeLeaf(tree, row).mu) for row in holdout], key=lambda x: x[1])
    ii = 0
    while row_mu_sorted[ii][1] == row_mu_sorted[ii+1][1]:
        if ii<len(row_mu_sorted)-1: ii += 1
        else: break
    ii = max(ii, the.Check)
    bests  = [r[0] for r in row_mu_sorted[:ii]]
    d2hs   = [disty(data,row) for row in bests]
    # [print(f"{win(k):.2f}", end=", ") for k in sorted(d2hs)]
    print("\nd2h\twin\tmu")
    for i,j,l in zip(d2hs, [win(kk) for kk in d2hs],[r[1] for r in row_mu_sorted[:ii]]):
      print(f"{i:.2f}\t{j:.2f}\t{win(l):.2f}")
    print("\n\n")
    # input()
  
  return asIs, ezr, causal 



def example():
  print("Example:")
  a = [1,2,1,1,3]
  b = [4,5,4,4,4]
  print(f"X\t,\tY")
  print("--------------------")
  for aa, bb in zip(a, b): print(f"{aa}\t~\t{bb}")
  print(f"\nMutual info = {mi(a,b):.2f}(high)\nCausality = X->Y is True and Y->X is False!")
  print(f"H(X|Y): {hcond(a,b):.2f},\tH(Y|X): {hcond(b,a):.2f}")
  print(f"H(X): {h(a):.2f},\tH(Y): {h(b):.2f}")
  input()
  print(f"X\t,\tY\t (Adding 2 rows to X and Y)")
  print("--------------------")
  for aa, bb in zip(a, b): print(f"{aa}\t~\t{bb}")
  a = [1,2,1,1,3, 6, 7]
  b = [4,5,4,4,4, 5, 4]
  print(f"{6}\t*\t{5}")
  print(f"{7}\t*\t{4}")
  print(f"\nMutual info = {mi(a,b):.2f}(high)\nCausality= X->Y is True and Y->X is False!")
  print(f"H(X|Y): {hcond(a,b):.2f},\tH(Y|X): {hcond(b,a):.2f}")
  print(f"H(X): {h(a):.2f},\tH(Y): {h(b):.2f}")
  input()

if __name__ == "__main__":
  # example()
  # play()

  filename = "../Data/moot/auto93.csv"
  data = Data(csv(filename))
  asIs, ezr, causal = [], [], []
  for _ in range(20):
    the.seed += 1
    asIs_, ezr_, causal_ = compareTrees(data)
    asIs.append(asIs_)
    ezr.append(ezr_)
    causal.append(causal_)
  

  rxs = {
      "asIs": asIs,
      "ezr": ezr,
      "causal": causal,
  }
  print("------ Distributions -------")
  for i,j in rxs.items():
      print(f"{i}:\t{sorted(j, reverse=True)}")

  print("------ New Stats -------")
  winners = stats2.top(rxs, reverse=True, Ks=0.90, Delta="medium")
  print("Top group :", winners)
  
  treatments = []
  print("------ Old Stats -------")
  asIs_some = stats.SOME(txt=f"asIs")
  asIs_some.adds(asIs)
  
  ezr_some = stats.SOME(txt=f"ezr")
  ezr_some.adds(ezr)
  
  causal_some = stats.SOME(txt=f"causal")
  causal_some.adds(causal)

  treatments.append(asIs_some)
  treatments.append(ezr_some)
  treatments.append(causal_some)

  stats.report(treatments)    
