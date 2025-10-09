from pathlib import Path
import sys
from typing import final

from pygam.pygam import none
from ezrSrc.ezr import *


def find_comment_lines(results_dir: Path) -> int:
    """Scan all CSV files under results_dir and print lines starting with '#'.

    Returns the number of matching lines printed.
    """
    nums = []
    try:
        with open(results_dir) as f:
            for line in f:
                if line.lstrip().startswith("#"):
                    nums.append(float(line.split(":")[-1].strip()))
    except Exception as e:
        print(f"Warning: failed to read : {e}", file=sys.stderr)
    return nums

def find_wins(nums, data = None):
    wins = [0]*len(nums)

    if data is None : data = Data(csv(f"Data/pt_data/{sys.argv[1]}"))
    b4   = adds(disty(data,row) for row in data.rows)
    win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
    best = lambda rows: win(disty(data, distysort(data,rows)[0]))
    idx = data.cols.y[0].at
    for r in data.rows:
        for ii, val in enumerate(nums):
            if val == r[idx]:
                wins[ii] = win(disty(data, r))
        if 0 not in wins:
            break
    return wins


def main() -> None:
    target_dir = "PromiseTune/results/" + sys.argv[1]
    wins = find_wins( find_comment_lines(target_dir) )
    print(wins)




if __name__ == "__main__":
    main()


