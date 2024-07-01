from gurobipy import *

point_set = [0, 1, 2]
arc_set = [0, 1]


model = Model()
x = {}
for i in range(len(arc_set)):
    x[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name="x_"+str(i))

model.update()

model.setObjective(x[0] + x[1], GRB.MINIMIZE)


model.addConstr(x[0] == 1)
model.addConstr(x[1] == x[0])

model.setParam('OutputFlag', 1)
model.optimize()

print(model.getVars())

