import numpy as np
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt


def iqr(x, a=0):
    return np.subtract(*np.percentile(x, [75, 25], axis=a))


if __name__ == '__main__':
    clusters = 3
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 0, 3]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))
    plt.show()
    #print(mix)

    clusters = [2, 3, 4]
    models = []
    idatas = []
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(mix.min(), mix.max(), cluster),
                              sd=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)

            sd = pm.HalfNormal('sd', sd=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
            idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)
    """
    ppc_mm = [pm.sample_posterior_predictive(idatas[i], 1000, models[i])
              for i in range(4)]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, constrained_layout=True)
    ax = np.ravel(ax)

    T_obs = iqr(mix)
    for idx, d_sim in enumerate(ppc_mm):
        T_sim = iqr(d_sim['y'][:100].T, 1)
    p_value = np.mean(T_sim >= T_obs)
    az.plot_kde(T_sim, ax=ax[idx])
    ax[idx].axvline(T_obs, 0, 1, color='k', ls='--')
    ax[idx].set_title(f'K = {clusters[idx]} \n p-value {p_value:.2f}')
    ax[idx].set_yticks([])
    """

    waic_l = az.waic(idatas, scale="deviance")
    loo_l = az.loo(idatas, scale="deviance")
    print(waic_l)
    print(loo_l)

    # WAIC
    comp = az.compare(dict(zip([str(c) for c in clusters], idatas)),
                      method='BB-pseudo-BMA', ic="waic", scale="deviance")
