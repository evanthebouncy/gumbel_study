import numpy as np

from numpy.random import gumbel
N = 10

logits = np.random.uniform(-2, 2, size=N)

def exp_sum_sample(x):
    exp_x = np.exp(x)
    exp_sum = np.sum(exp_x)
    return np.random.choice([i for i in range(N)], p=exp_x/exp_sum)

def gumbel_max_sample(x):
    z = gumbel(loc=0, scale=1, size=x.shape)
    return (x + z).argmax()

def collect_sample(sampler, logits, n):
    ret = dict()
    for i in range(n):
        sampl = sampler(logits)
        if sampl not in ret:
            ret[sampl] = 0
        ret[sampl] += 1
    return ret

print (collect_sample(exp_sum_sample, logits, 10000))
print (collect_sample(gumbel_max_sample, logits, 10000))
