from audioop import reverse
from ezr import *
import matplotlib.pyplot as plt
import numpy as np

def create_freq_chart(filename):
  data = Data(csv(filename))
  b4   = adds(disty(data,row) for row in data.rows)
  data.rows = shuffle(data.rows)

  win  = lambda v: int(100*(1 - (v - b4.lo)/(b4.mu - b4.lo)))
  best = lambda rows: win(disty(data, distysort(data,rows)[0]))

  distances = [disty(data, r) for r in data.rows]
  wins = [win(disty(data, r)) for r in data.rows]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
      
  # Sort distances for CDF
  sorted_distances = np.sort(distances,)
  n = len(sorted_distances)
  cumulative_prob = np.arange(1, n + 1) / n
  
  # Plot CDF
  ax1.plot(sorted_distances, cumulative_prob, linewidth=2, color='blue', label='CDF')
  ax1.set_xlabel('D2h Values')
  ax1.set_ylabel('Frequency')
  ax1.grid(True, alpha=0.3)
  ax1.set_ylim(0, 1)
  
  mean_val = np.mean(distances)
  std_val = np.std(distances)
  median_val = np.median(distances)
  ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
  ax1.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
  ax1.legend()

  # Box plot
  ax2.hist(wins, bins=1000, alpha=0.7, color='skyblue', edgecolor='black')
  ax2.set_xlabel('Win() Values')
  ax2.grid(True, alpha=0.3)
  mean_val = np.mean(wins)
  std_val = np.std(wins)
  median_val = np.median(wins)
  ax2.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
  ax2.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
  ax2.legend()

  # Add statistics summary
  stats_text = f"""Statistics Summary:
  Count: {len(distances)}
  Mean: {mean_val:.3f}
  Std: {std_val:.3f}
  Min: {min(distances):.3f}
  Max: {max(distances):.3f}
  Median: {median_val:.3f}"""

  ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

  plt.tight_layout()
  plt.savefig(f"data_dist/{filename.split("/")[-1][:-4]}", dpi=300, bbox_inches='tight')

def main():
  systems = ['LLVM', '7z', 'BDBC', 'exastencils', 'dconvert', 'deeparch',
            'PostgreSQL', 'javagc', 'storm', 'x264',
            'redis', 'HSQLDB']
  for system in systems:
    filename = "../Data/pt_data/" + system + ".csv"
    create_freq_chart(filename)



if __name__ == "__main__":
    main()