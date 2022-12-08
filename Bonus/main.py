import numpy as np
import pymc3 as pm

if __name__ == '__main__':

    # Punctul 1
    statii_gatit = 3
    statii_comanda = 2

    trafic = np.random.poisson(20, 70)

    # Punctul 2
    mese = 5

    timp_gatit = 0
    timp_comanda = 0
    timp_mancat = 0
    contor = 0
    contor_mese = 0
    for i in range(len(trafic)):
        model = pm.Model()
        with model:
            comanda = pm.Normal('N', sigma=0.5, mu=1)
            gatit = pm.Exponential('G', 1 / 2)
            mancat = pm.Normal('M', mu=10, sigma=2)
        trace = pm.sample(trafic[i], chains=1, model=model)
        dictionary = {
            'comanda': trace['N'].tolist(),
            'gatit': trace['G'].tolist(),
            'mancat': trace['M'].tolist()
        }

        timp_comanda = sum(dictionary['comanda']) / statii_comanda
        timp_gatit = sum(dictionary['gatit']) / statii_gatit
        timp_mancat = sum(dictionary['mancat']) / mese
        if timp_gatit + timp_comanda <= 60:
            contor += 1
        if timp_mancat <= 60:
            contor_mese += 1
    print(contor / 100)
    print(contor_mese / 100)
