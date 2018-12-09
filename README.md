# Facility-Location
Facility Location code for COMPSCI 690AA: Approximation Algorithms

Pre-requisites: Numpy, Scipy and [PuLP](https://pypi.org/project/PuLP/)

To use this code:

 1. import facility
 2. Create a facility location problem instance by `F = Facility_Location(facility_costs,demand_costs)`. Here, `facility_costs` are the costs for setting up facilities and `demand_costs` are the costs for a facility to service a particular client. `demand_costs[j][i]` is the cost of facility `i` to service client `j`.
 3. Call `F.solve_LP()`. This does not return anything. It just solves the LP and gets the primal and dual variables.
 4. Call `F.Deterministic_rounding()` to use deterministic rounding and `F.Random_rounding()`for random rounding. These return a dictionary which represents the assignment.
 5. Use `F.cost(assignment)`to get the cost of an assignment.
 6. `F.ILP()`solves the problem exactly, using integer linear programming. It returns the optimal assignment. It can be used to compare the rounded solutions with the optimal one.
