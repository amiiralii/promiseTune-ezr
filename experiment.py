from ezrSrc.ezr import *
import random
import os
import ezrSrc.stats as stats
import ezrSrc.stats2 as stats2
from extractor import *

def main():
    filename = sys.argv[1]
    data = Data(csv(filename))
    half = len(data.rows) // 2
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

    # cols = [d.txt for d in data.cols.all]
    # targets = [c for c in cols if c[-1] in ["+", "-"]]
    # features = [c for c in cols if c[-1] not in ["+", "-", "X"]]
    
    repeats = 8
    picks = 10
    treatments = []

    print("EZR1 Labeling Budget,", picks + stp)
    print("EZR2 Labeling Budget,", 2*(picks + stp))
    print("---------------------")
    asIs, ezr1, ezr2 = [], [], []
    ezr2_counter = 0
    ezr2_exec_counter = 0
    for _ in range(repeats):
        the.seed += 1
        random.seed += 1
        the.acq = "near"

        ### asIs treatment
        idx = random.randrange(len(data.rows))
        selected_win_rate = win(disty(data, data.rows[idx]))
        selected_opt_val  = data.rows[idx][-1]
        asIs.append(selected_win_rate)


        ### EZR treatment
        labels = likely(data)
        tree = Tree(clone(data, labels))
        # treeShow(data, tree, win)
        row_mu_sorted = sorted([(row, treeLeaf(tree, row).mu) for row in data.rows], key=lambda x: x[1])
        ii = 0
        while row_mu_sorted[ii][1] == row_mu_sorted[ii+1][1]:
            if ii<len(row_mu_sorted)-1: ii += 1
            else: break
        # print("How many falls under best branch?", ii,"out of", len(row_mu_sorted))
        selection = random.sample(row_mu_sorted[:ii], min(ii,picks))
        best_win_rate = best([i[0] for i in selection])
        best_opt_val = sorted(selection, key = lambda r: r[0][-1])[0][0][-1]
        # print(f"Selected win rate: {best_win_rate}, Selected opt val: {best_opt_val}")
        ezr1.append(best_win_rate)

        sorted_sub_data = sorted([rows[0] for rows in row_mu_sorted[:ii]], key = lambda x: x[-1])
        rnk_1=0
        for ssd in sorted_sub_data:
            if ssd[-1] > best_opt_val:
                # print("Rank = ",rnk_1, "out of", ii)
                break
            rnk_1 += 1

        if ii > stp:
            the.acq = "near"
            sub_data = clone(data, [i[0] for i in row_mu_sorted[:ii]])
            sub_labels = likely(sub_data)
            sub_tree = Tree(clone(sub_data, sub_labels))
            # treeShow(data, sub_tree, win)
            sub_row_mu_sorted = sorted([(row, treeLeaf(sub_tree, row).mu) for row in sub_data.rows], key=lambda x: x[1])
            iii = 0
            while sub_row_mu_sorted[iii][1] == sub_row_mu_sorted[iii+1][1]:
                if iii<len(sub_row_mu_sorted)-2: iii += 1
                else: break
            # print("How many falls under best sub_branch?", ii,"out of", len(sub_row_mu_sorted))
            sub_selection = random.sample(sub_row_mu_sorted[:ii], min(ii,picks))
            sub_best_win_rate = best([i[0] for i in sub_selection])
            sub_best_opt_val = sorted(sub_selection, key = lambda r: r[0][-1])[0][0][-1]
            # print(f"Selected win rate: {sub_best_win_rate}, Selected opt val: {sub_best_opt_val}")
            ezr2.append(sub_best_win_rate)
            sorted_sub_data = sorted(sub_data.rows, key = lambda x: x[-1])
            rnk_2=0
            for ssd in sorted_sub_data:
                if ssd[-1] > sub_best_opt_val:
                    # print("Rank = ",rnk_2, "out of", len(sub_data.rows))
                    break
                rnk_2 += 1
        else:
            ezr2.append(best_win_rate) 
            rnk_2 = -1
        print(f"Ranks--> EZR-1: {rnk_1}\tEZR-2 {rnk_2}\tOut of {ii}\t/ {len(data.rows)}")
        if rnk_2 < rnk_1 and rnk_2 != -1: ezr2_counter += 1
        if rnk_2 != -1: ezr2_exec_counter += 1

    print(f"Ezr2 is better in {ezr2_counter} cases out of {ezr2_exec_counter}.")

    rxs = {
        "asIs": asIs,
        "EZR1": ezr1,
        "EZR2": ezr2,
        "PT"  : find_wins( find_comment_lines(f"PromiseTune/results/{sys.argv[1].split("/")[-1]}"), data )
    }
    print("------ Distributions -------")
    for i,j in rxs.items():
        print(f"{i}:\t{j}")

    print("------ New Stats -------")
    winners = stats2.top(rxs, reverse=True, Ks=0.90, Delta="medium")
    print("Top group :", winners)
    

    print("------ Old Stats -------")
    asIs_some = stats.SOME(txt=f"asIs")
    asIs_some.adds(asIs)
    treatments.append(asIs_some)
    ezr1_some = stats.SOME(txt=f"EZR1")
    ezr1_some.adds(ezr1)
    treatments.append(ezr1_some)
    ezr2_some = stats.SOME(txt=f"EZR2")
    ezr2_some.adds(ezr2)
    treatments.append(ezr2_some)

    pt_some = stats.SOME(txt=f"PT")
    pt_some.adds(find_wins( find_comment_lines(f"PromiseTune/results/{sys.argv[1].split("/")[-1]}"), data ))
    treatments.append(pt_some)

    stats.report(treatments)        

if __name__ == "__main__":
    main()