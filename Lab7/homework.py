import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import arviz as az
import math

if __name__ == '__main__':
    data = pd.read_csv("Prices.csv")
    # Ex1
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta1 = pm.Normal('beta1', mu=0, sd=1)
        beta2 = pm.Normal('beta2', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        mu = pm.Deterministic('mu', alpha + beta1 * data['Speed'].values + beta2 * [math.log(i) for i in
                                                                                    data['HardDrive'].values])
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'].values)
        trace = pm.sample(5000, tune=1000, chains=1)


    # Ex2
    az.plot_posterior({"beta1": trace['beta1'], "beta2": trace['beta2']}, hdi_prob=0.95)

    """Ex3
    Din datele de la # Ex2, se observa ca beta1 si beta 2 sunt diferite de 0 (in medie) si au o deviatie standard 
    acceptabila,deci acestia sunt indicatori utili pentru pret. 
    """

    # Ex4
    az.plot_posterior({"Pret_asteptat": trace['mu']}, hdi_prob=0.9)


    # bonus
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        beta = pm.Normal('beta', mu=0, sd=1)
        sigma = pm.HalfNormal('sigma', sd=1)
        mu = pm.Deterministic('mu', alpha + beta * [0 if i == "no" else 1 for i in data['Premium'].values])
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'].values)
        trace = pm.sample(5000, tune=1000, chains=1)
    az.plot_posterior({"premium_beta": trace['beta']})
    plt.show()
# Din bonus.png se observa ca daca producatorul este premium are un impact mare (media beta este aproape 0)
