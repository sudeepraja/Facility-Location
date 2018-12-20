import numpy as np
from scipy.linalg import norm
from pulp import *

class Facility_Location(object):
	"""Facility Location Class
	Arguments:
		facility_costs: cost of setting up facilities
		demand_costs: matrix of demand costs
	"""
	def __init__(self, facility_costs, demand_costs):
		self.facility_costs = facility_costs
		self.demand_costs = demand_costs
		self.num_facilities = len(facility_costs)
		self.num_demands = len(demand_costs)

		assert self.num_facilities == np.shape(self.demand_costs)[1], "Shape mismatch in facilities and demand"

	def cost(self, assignment):
		cost = 0
		for i in assignment:
			cost+= self.facility_costs[i]
			for j in assignment[i]:
				cost+= self.demand_costs[j][i]
		return cost

	def ILP(self):
		primal = LpProblem("Facility Location Primal",LpMinimize)

		y = [LpVariable("y"+str(i),cat='Binary') for i in range(self.num_facilities)]
		x = [[LpVariable("x"+str(i)+str(j),cat='Binary') for i in range(self.num_facilities)] for j in range(self.num_demands)]

		c_1 = sum(y[i]*self.facility_costs[i] for i in range(self.num_facilities))
		c_2 = sum(x[j][i]*self.demand_costs[j][i] for i in range(self.num_facilities) for j in range(self.num_demands))

		primal += c_1+c_2

		for j in range(self.num_demands):
			primal += sum(x[j][i] for i in range(self.num_facilities)) == 1

		for i in range(self.num_facilities):
			for j in range(self.num_demands):
				primal += x[j][i] - y[i] <= 0

		primal.solve()
		primals = {a.name:a.varValue for a in primal.variables()}
		y = [primals["y"+str(i)] for i in range(self.num_facilities)]
		x = [[primals["x"+str(i)+str(j)] for i in range(self.num_facilities)] for j in range(self.num_demands)]

		Facilities_chosen = {i:{j for j in range(self.num_demands) if x[j][i]==1.0} for i in range(self.num_facilities) if y[i]==1.0}
		return Facilities_chosen


	def solve_LP(self):

		## Create the Primal Problem
		primal = LpProblem("Facility Location Primal",LpMinimize)

		y = [LpVariable("y"+str(i),0,None) for i in range(self.num_facilities)]
		x = [[LpVariable("x"+str(i)+str(j),0,None) for i in range(self.num_facilities)] for j in range(self.num_demands)]

		c_1 = sum(y[i]*self.facility_costs[i] for i in range(self.num_facilities))
		c_2 = sum(x[j][i]*self.demand_costs[j][i] for i in range(self.num_facilities) for j in range(self.num_demands))

		primal += c_1+c_2

		for j in range(self.num_demands):
			primal += sum(x[j][i] for i in range(self.num_facilities)) == 1

		for i in range(self.num_facilities):
			for j in range(self.num_demands):
				primal += x[j][i] - y[i] <= 0

		primal.solve()
		# print primal
		# for v in primal.variables():
		# 	print v.name, "=", v.varValue 
		# print value(primal.objective)

		## Create the Dual Problem
		dual = LpProblem("Facility Location Dual",LpMaximize)

		v = [LpVariable("v"+str(j),None,None) for j in range(self.num_demands)]
		w = [[LpVariable("w"+str(i)+str(j),0,None) for i in range(self.num_facilities)] for j in range(self.num_demands)]

		dual += sum(v)
		for i in range(self.num_facilities):
			dual+= sum(w[j][i] for j in range(self.num_demands)) <= self.facility_costs[i]

		for i in range(self.num_facilities):
			for j in range(self.num_demands):
				dual += v[j] - w[j][i] <= self.demand_costs[j][i]

		dual.solve()
		# print dual
		# for v in dual.variables():
		# 	print v.name, "=", v.varValue 
		# print value(dual.objective)

		self.primal = primal
		self.dual = dual

		primals = {a.name:a.varValue for a in self.primal.variables()}
		duals = {a.name:a.varValue for a in self.dual.variables()}


		## Get the primal and dual variables
		self.y = [primals["y"+str(i)] for i in range(self.num_facilities)]
		self.x = [[primals["x"+str(i)+str(j)] for i in range(self.num_facilities)] for j in range(self.num_demands)]
		self.v = [duals["v"+str(j)] for j in range(self.num_demands)]
		self.w = [[duals["w"+str(i)+str(j)] for i in range(self.num_facilities)] for j in range(self.num_demands)]

		## Find the Neighbors of demand nodes
		self.Neighbors = {j:{i for i in range(self.num_facilities) if self.x[j][i]>0.0} for j in range(self.num_demands)}
		
		## Find the second neighbors of demand nodes
		self.Neighbors_2 = {j:{k for k in range(self.num_demands) if len(self.Neighbors[j].intersection(self.Neighbors[k]))!=0 } for j in range(self.num_demands)}

	def Deterministic_rounding(self):

		C = set(range(self.num_demands))
		k = 0

		Facilities_chosen=dict()

		while len(C)!=0:
			k+=1

			## Choose j_k that minimizes v*[j]+C*[j] over all j in C
			j_k = min({(self.v[j],j) for j in C})[1]

			## Choose i_k to be the cheapest facility in  Neighbors of j_k
			i_k = min({(self.facility_costs[i],i) for i in self.Neighbors[j_k]})[1]

			## Create facility and remove demands from C
			Facilities_chosen[i_k] = {j_k}.union(C.intersection(self.Neighbors_2[j_k]))
			C = C.difference({j_k}.union(self.Neighbors_2[j_k])) 

		return Facilities_chosen

	def Random_rounding(self):

		C = set(range(self.num_demands))
		k = 0

		C_j = [sum([self.x[j][i]*self.demand_costs[j][i] for i in range(self.num_facilities)]) for j in range(self.num_demands)]

		Facilities_chosen = dict()
		while len(C)!=0:
			k+=1

			## Choose j_k that minimizes v*[j]+C*[j] over all j in C
			j_k = min({(self.v[j]+C_j[j],j) for j in C})[1]

			## Choose facility i_k by random sampling according to distribution x*[j_k]
			i_k = np.random.choice(range(self.num_facilities),p=self.x[j_k])

			## Create facility and remove demands from C
			Facilities_chosen[i_k] = {j_k}.union(C.intersection(self.Neighbors_2[j_k]))
			C = C.difference({j_k}.union(self.Neighbors_2[j_k])) 

		return Facilities_chosen





def main():
	np.random.seed(3)
	f=3
	d=8

	facilities_matrix = np.random.randint(0,100, size=(f,3))
	demands_matrix = np.random.randint(0,100, size=(d,3))

	print "Facilities points:",f,"\n", facilities_matrix
	print "Demand points:",d,"\n",demands_matrix

	## Give a list of facility costs
	facility_costs = np.random.randint(low = 1, high = 100, size = f)

	## Give array of demand costs. Indexed by demand, facility.
	demand_costs = np.zeros((d,f))

	for i in range(f):
		for j in range(d):
			demand_costs[j][i] = norm(demands_matrix[j]-facilities_matrix[i])

	print "Cost matrix:\n",demand_costs

	## Create Instance
	F = Facility_Location(facility_costs,demand_costs)

	## Solve primal and dual
	F.solve_LP()

	A = F.Deterministic_rounding()
	B = F.Random_rounding()
	C = F.ILP()

	print "\n\n"

	print "Exact Solution by Integer Linear Program"
	print C
	print "cost:",F.cost(C)

	print "\n\n"

	print "Deterministic Rounding"
	print A
	print "cost:",F.cost(A)

	print "\n\n"

	print "Random Rounding"
	print B
	print "cost:",F.cost(B)

	print "\n\n"

if __name__ == '__main__':
	main()
