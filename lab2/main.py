import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az


#ex1
x = stats.expon.rvs(scale=1 / 4, size=1000)
y = stats.expon.rvs(scale=1 / 6, size=1000)
z = stats.binom.rvs(1, 0.4, size=1000)

q = []
for i in range(1, 1000):
    if (z[i] == 1):
        q.append(x[i])
    else:
        q.append(y[i])

az.plot_posterior({'q': q})
plt.show()
print("media este:", np.mean(q))
print("deviatia standard:", np.std(q))


#ex2
# latency = stats.expon.rvs(0, 1 / 4, size=SAMPLE_SIZE_FOR_EACH_SERVER)

server1 = stats.gamma.rvs(4, scale=1 / 3, size=int(10000 * 0.25)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.25))
server2 = stats.gamma.rvs(4, scale=1 / 2, size=int(10000 * 0.25)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.25))
server3 = stats.gamma.rvs(5, scale=1 / 2, size=int(10000 * 0.3)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.3))
server4 = stats.gamma.rvs(5, scale=1 / 3, size=int(10000 * 0.2)) + stats.expon.rvs(0, 1 / 4, size = int(10000 * 0.2))

avg = np.concatenate([server1, server2, server3, server4])
favorable = 0
for i in avg:
    if i > 3:
        favorable += 1

print("Probability: ", favorable/10000)

az.plot_posterior({"Avg:": avg})
plt.show()

# ex3

#  A si B sunt  independente => p(A ^ B) = p(A) * p(B)
# A = np.random.choice(a=['s', 'b'], p=[0.5, 0.5], size=1)
# B = np.random.choice(a=['s', 'b'], p=[0.3, 0.7], size=1)
ss = []
sb = []
bs = []
bb = []
for i in range(0, 100):
    rez = []
    for i in range(0, 10):
        A = np.random.choice(a=['s', 'b'], p=[0.5, 0.5], size=1)[0]
        B = np.random.choice(a=['s', 'b'], p=[0.3, 0.7], size=1)[0]
        match A, B:
            case 's', 's':
                rez.append('ss')
            case 's', 'b':
                rez.append('sb')
            case 'b', 's':
                rez.append('bs')
            case 'b', 'b':
                rez.append('bb')
    ss.append(rez.count('ss'))
    sb.append(rez.count('sb'))
    bs.append(rez.count('bs'))
    bb.append(rez.count('bb'))

az.plot_posterior({"S, S": np.array(ss)})
az.plot_posterior({"S, B": np.array(sb)})
az.plot_posterior({"B, S": np.array(bs)})
az.plot_posterior({"B, B": np.array(bb)})
plt.show()

