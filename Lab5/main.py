from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

bet_model = BayesianNetwork(
    [
        ("c1", "c2"),
        ("c1", "p1"),
        ("c1", "p3"),
        ("c2", "p2"),
        ("p1", "p2"),
        ("p1", "p3"),
        ("p2", "p3")
    ]
)

CPD_c1 = TabularCPD(variable='c1', variable_card=5, values=[[1/5], [1/5], [1/5], [1/5], [1/5]])
#print(CPD_c1)

CPD_c2 = TabularCPD(
    variable='c2',
    variable_card=5,
    values=[[0, 1/4, 1/4, 1/4, 1/4], [1/4, 0, 1/4, 1/4, 1/4], [1/4, 1/4, 0, 1/4, 1/4], [1/4, 1/4, 1/4, 0, 1/4], [1/4, 1/4, 1/4, 1/4, 0]],
    evidence=['c1'],
    evidence_card=[5]
)
#print(CPD_c2)

CPD_p1 = TabularCPD(
    variable='p1',
    variable_card=2,
    values=[[0.99, 0.8, 0.75, 0.2, 0.1], [0.01, 0.2, 0.25, 0.8, 0.9]],
    evidence=['c1'],
    evidence_card=[5]
)
CPD_p2 = TabularCPD(
    variable='p2',
    variable_card=3,
    values=[[0.33,0.33,0.33], [0.33,0.33,0.33], [0.33, 0.33, 0.33]],
    evidence=['c1', 'p1'],
    evidence_card=[3, 1]
)

bet_model.add_cpds(
    CPD_c1, CPD_c2,  CPD_p1
)
