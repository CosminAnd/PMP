import math
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import arviz as az


if __name__ == '__main__':
    data = pd.read_csv('Admission.csv')
    df = data
    y1 = pd.Categorical(df['Admission']).codes

    x_n = ['GRE', 'GPA']
    x1_ax = df[x_n].values

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))
        μ = alpha + pm.math.dot(x1_ax, beta)

        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-μ)))
        bd = pm.Deterministic('bd', -alpha / beta[1] - beta[0] / beta[1] * x1_ax[:, 0])
        yl = pm.Bernoulli('yl', p=theta, observed=y1)
        idata_1 = pm.sample(2000, target_accept=0.9, return_inferencedata=True)

    # granita de decizie
    sorted_idx = np.argsort(x1_ax[:, 0])
    bd = idata_1.posterior['bd'].mean(("chain", "draw"))[sorted_idx]
    plt.scatter(x1_ax[:, 0], x1_ax[:, 1], c=[f'C{x}' for x in y1])
    plt.plot(x1_ax[:, 0][sorted_idx], bd, color='k')

    # grafica
    az.plot_hdi(x1_ax[:, 0], idata_1.posterior['bd'], color='k')
    plt.xlabel(x_n[0])
    plt.ylabel(x_n[1])

