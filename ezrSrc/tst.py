from stats2 import *
import random

x = []
y = [random.gauss(0.2, 1.0) for _ in range(200)]


rxs = {
  "A": [random.gauss(3.0, 1.0) for _ in range(200)],
  "B": [random.gauss(3.0, 1.0) for _ in range(200)],
  "C": [random.gauss(4.0, 1.0) for _ in range(200)],
  "D": [random.gauss(5.0, 1.0) for _ in range(200)],
}

# Choose reverse=False if smaller is better (e.g., time, error). Use reverse=True if larger is better.
winners = top(rxs, reverse=False, Ks=0.95, Delta="smed")
print("Top group:", winners)