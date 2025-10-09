from fileinput import filename
from ezrSrc.ezr import *
import random
import os
import ezrSrc.stats as stats
from extractor import *

def main():
    systems = ['LLVM', '7z', 'BDBC', 'exastencils', 'dconvert', 'deeparch',
            'PostgreSQL', 'javagc', 'storm', 'x264',
            'redis', 'HSQLDB']
    t0 = time.time()
    for system in systems:
        t1 = time.time()
        filename = "Data/pt_data/" + system + ".csv"
        data = Data(csv(filename))
        b4   = adds(disty(data,row) for row in data.rows)
        data.rows = shuffle(data.rows)
        # train, holdout = clone(data, data.rows[:half]), clone(data, data.rows[half:])
        win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
        best = lambda rows: win(disty(data, distysort(data,rows)[0]))

        seed = 1
        stp = 50
        the.Budget = stp
        the.acq = "near"
        the.few = 256
        the.Any = 5
        the.seed = seed
        random.seed = seed

        picks = 10

        ### EZR treatment
        labels = likely(data)
        tree = Tree(clone(data, labels))
        # treeShow(data, tree, win)
        row_mu_sorted = sorted([(row, treeLeaf(tree, row).mu) for row in data.rows], key=lambda x: x[1])
        ii = 0
        while row_mu_sorted[ii][1] == row_mu_sorted[ii+1][1]:
            if ii<len(row_mu_sorted)-1: ii += 1
            else: break

        selection = random.sample(row_mu_sorted[:ii], min(ii,picks))
        best_win_rate = best([i[0] for i in selection])
        best_opt_val = sorted(selection, key = lambda r: r[0][-1])[0][0][-1]


        sorted_sub_data = sorted([rows[0] for rows in row_mu_sorted[:ii]], key = lambda x: x[-1])
        rnk_1=0
        for ssd in sorted_sub_data:
            if ssd[-1] > best_opt_val:
                # print("Rank = ",rnk_1, "out of", ii)
                break
            rnk_1 += 1

        print(f"Time {system}.csv : {time.time()-t1:.2f}")
    print(f"Time total : {time.time()-t0:.2f}")

if __name__ == "__main__":
    main()