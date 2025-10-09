import math
def Z(x, mu, sd):
    return (x-mu) / sd + 1e-32

def cdf(x, mu, sd):
    z = Z(x, mu, sd)
    return 1/(1+(math.e ** (-4*math.pi*z/(9-z))))


for i in range(1,20):
    print(i, cdf(i, 10, 2))