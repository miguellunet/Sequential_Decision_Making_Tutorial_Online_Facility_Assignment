from miscellaneous import distance_calculator
from gurobipy import GRB, Model
import numpy as np

# Fixes Gurobi's own internal tie-breaking/search order
# MIP with multiple equally-optimal solutions returns the same one every run.
GUROBI_SEED = 42

# May be used to set a time limit for the optimization, but is commented out to allow the solver to run without a time limit.
#TIME_LIMIT_SECONDS = 1

def solve_milp_lp(instance, warehouses_location, warehouses_initial_capacity, num_customers, divisible = False, divisible_except_first_one = False):

    # This function is used to solve all the possible lookahead models in the paper: PEL, DEL, PLA, LPH.
    # When provided with the perfect hindsight instance, it solves the offline MIP (Eqs. 1-4) to get the optimal assignment of customers to warehouses.
    # The differences between the policies are in the way the model is formulated (binary vs continuous) and in the way the instance is generated (random sampling vs region-based sampling).

    # Description of possible model variants:
    # divisible = True: the model is formulated as a continuous LP (used for LPH and LPE)
    # divisible = False: the model is formulated as a binary MIP (used for PHS and PEL)
    # divisible_except_first_one = True: the first customer is forced to be assigned to a single warehouse (binary), while the rest of the customers can be assigned fractionally (continuous). This is used for DEL and PLA, which have a binary assignment for the first customer and continuous assignments for the rest.

    # Get the list of all customers in the instance
    all_customers = instance[:num_customers]

    # Sets
    T = np.arange(len(all_customers)) # Set of customers
    F = np.arange(len(warehouses_initial_capacity)) # Set of warehouses

    # Parameters
    d = np.array([[distance_calculator(customer[1:3], warehouses_location[f]) for f in F] for customer in all_customers])   # distance from customer t to warehouse f
    c = warehouses_initial_capacity     # capacity of warehouse f
    s = np.array([customer[3] for customer in all_customers])   # demand of customer t

    # Create a new model
    m = Model("assignment")

    if divisible:
        # Create variables
        z = m.addVars(T, F, vtype=GRB.CONTINUOUS, name="z")

        if divisible_except_first_one:
            # Make the first row binary
            for f in F:
                z[0, f].vtype = GRB.BINARY

        # Set objective
        m.setObjective(sum(z[t,f]*d[t,f] for t in T for f in F), GRB.MINIMIZE)

        # Add constraints
        m.addConstrs((sum(z[t,f] for f in F) == s[t] for t in T), "c1") #demand constraint
        m.addConstrs((sum(z[t,f] for t in T) <= c[f] for f in F), "c2") #capacity constraint

    else:
        # Create variables
        z = m.addVars(T, F, vtype=GRB.BINARY, name="z")

        # Set objective
        m.setObjective(sum(z[t,f]*d[t,f] for t in T for f in F), GRB.MINIMIZE)

        # Add constraints
        m.addConstrs((sum(z[t,f] for f in F) == 1 for t in T), "c1") # demand constraint
        m.addConstrs((sum(z[t,f] for t in T) <= c[f] for f in F), "c2") # capacity constraint

    # Make the model quiet
    m.setParam('OutputFlag', 0)
    m.setParam('Seed', GUROBI_SEED)

    # Can be used to set a time limit for the optimization, but it is commented out to allow the solver to run without a time limit.
    '''
    # Optimize model, capped at TIME_LIMIT_SECONDS
    m.setParam('TimeLimit', TIME_LIMIT_SECONDS)
    m.optimize()

    if m.SolCount == 0:
        # no incumbent within the time limit -> keep solving with no time limit, but
        # stop as soon as the first feasible solution is found, so we always return
        # an assignment instead of failing on the .x/.ObjVal reads below
        m.setParam('TimeLimit', GRB.INFINITY)
        m.setParam('SolutionLimit', 1)
        m.optimize()
    '''

    # Optimize model
    m.optimize()

    #Get the assignments
    assignments = []
    for t in T:
        for f in F:
            if z[t,f].x > 0.5:
                assignments.append(f)
                break
    
    #Get the dual values from constraint (c2) -> used in the LPH policy
    if divisible and not divisible_except_first_one:    # all constraints are continuous, so we can get the dual values
        dual_values = [m.getConstrByName('c2['+str(f)+']').getAttr('Pi') for f in F]
    else:
        dual_values = None

    return assignments, m.ObjVal, dual_values