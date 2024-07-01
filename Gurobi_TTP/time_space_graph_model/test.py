from pyomo.environ import *

model = ConcreteModel()
model.x = Var()
model.y = Var()
model.objective = Objective(expr=model.x + 4 * model.y, sense=minimize)
model.constraint = Constraint(expr=model.x + model.y >= 1)

solver = SolverFactory('gurobi')
solver.solve(model)

print(value(model.x))
print(value(model.y))