import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

model = pm.Model()

with model:
    cutremur = pm.Bernoulli('C', 0.0005)
    incendiu_c = pm.Deterministic('I_c', cutremur, pm.math.switch(cutremur, 0.03, 0.01))
    incendiu = pm.Bernoulli('I', p=incendiu_c)
    alarma_c= pm.Deterministic("A_c", pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.02), pm.math.switch(incendiu, 0.95, 0.0001)))
    alarma = pm.Bernoulli('A', p=alarma_c)

trace = pm.sample(20000)

dictionary = {
    'cutremur': trace['C'].tolist(),
    'alarma': trace['A'].tolist(),
    'incendiu': trace['I'].tolist()
}
df = pd.DataFrame(dictionary)

p_cutremur= df[((df['cutremur']==1)& (df['alarma']==0))].shape[0]/ df[df['cutremur']==0].shape[0]
print(p_cutremur)

p_incendiu = df[((df['incendiu']==1)& (df['alarma']==0))].shape[0]/ df[df['alarma']==0].shape[0]
print(p_incendiu)

az.plot_posterior(trace)
plt.show()