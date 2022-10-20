import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()
alpha = 3

with model:
    rng = np.random.default_rng(12345)
    rints = rng.integers(low=0, high= 20)

    traffic = pm.Poisson("T", mu=20)
    for i in range(0, rints):
        order = pm.Normal("O"+str(i), 1, sigma=0.5)
        cook = pm.Exponential("C"+str(i), lam=alpha)
    trace = pm.sample(10000, chains=1)