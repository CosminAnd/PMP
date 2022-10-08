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
