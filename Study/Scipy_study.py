from scipy.optimize import minimize, LinearConstraint, milp
import numpy as np

c = [1, 1]
constraints = LinearConstraint(A=[[1, 0], [0, 1]], lb=[1, 2], ub=[np.inf, np.inf])
integrality = np.ones_like(c)

result = milp(c, constraints=constraints, integrality=integrality)
print(result)
