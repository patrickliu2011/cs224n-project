from pysat.examples.rc2 import RC2
from pysat.formula import WCNF

wcnf = WCNF()
wcnf.extend([[0], [-4], [-5], [-6]], weights=[1, 5, 3, 4])
print(wcnf.soft)
wcnf.append([-1, -2]) # adding hard clauses
wcnf.append([-1, -3])

wcnf.append([1], weight=1) # adding soft clauses
wcnf.append([2], weight=1)
wcnf.append([3], weight=1)

with RC2(wcnf) as rc2:
    rc2.compute() # solving the MaxSAT problem

    print(rc2.cost, rc2.model) # cost + solution
    rc2.add_clause([-2, -3]) # adding one more hard clause
    rc2.compute() # computing another model
    print(rc2.cost, rc2.model) # cost + solution

rc2.delete()