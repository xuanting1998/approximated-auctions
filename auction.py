import numpy as np
from scipy.optimize import linprog
import itertools as it
import time

NUM_APPROX_VALS = 200
MILLIS_PER_SEC = 1000

def time_method(method, arguments=[]):
    start = time.clock()
    result = method(*arguments)
    end = time.clock()
    return result + (MILLIS_PER_SEC * (end - start),)

class CostFn(object):
    def __init__(self, coefficients, capacity):
        self.fn = np.polynomial.Polynomial(coefficients)
        self.capacity = capacity

    def eval(self, values):
        return self.fn(values)

    def get_capacity(self):
        return self.capacity

    def get_coef(self):
        return self.fn.coef

# Used in nonlinear optimization over polytope
class Vertex(object):
    @staticmethod
    def is_valid(binding_constraints, constraints):
        A, b = Vertex.separate_constraints(constraints)
        return np.linalg.matrix_rank(A[binding_constraints,]) == A.shape[1]

    @staticmethod
    def separate_constraints(constraints):
        return constraints[:, :-1], constraints[:, -1]

    def __init__(self, intersecting_constraints, constraints):
        self.binding_constraints = intersecting_constraints
        self.coords = self.compute_coords_from_constraints(constraints)

    def get_coords(self):
        return self.coords

    def get_binding_constraints(self):
        return self.binding_constraints

    def compute_coords_from_constraints(self, constraints):
        A, b = Vertex.separate_constraints(constraints)
        return np.linalg.solve(A[self.get_binding_constraints(), ], b[self.get_binding_constraints(), ])

    def is_feasible(self, constraints):
        return self.first_invalid_constraint(constraints) >= constraints.shape[0]

    def first_invalid_constraint(self, constraints):
        A, b = Vertex.separate_constraints(constraints)
        Ax = np.dot(A, self.get_coords())

        if Ax[0] != b[0]:
            return 0

        constraints_not_satisfied = Ax[1:] > b[1:]
        if np.any(constraints_not_satisfied):
            return 1 + np.argmax(constraints_not_satisfied) # returns first index of invalid constraint

        return constraints.shape[0] # no invalid constraint was found so return index past last constraint

    def __hash__(self):
        return hash(self.get_binding_constraints())

    def __eq__(self, other):
        return self.get_binding_constraints() == other.get_binding_constraints()


class Auction(object):
    def __init__(self, quota, cost_fns, approx_method = 'least_squares'):
        self.quota = quota
        self.approx_method = approx_method

        self.cost_fns = cost_fns
        self.approx_cost_fns = self.approx_cost_fns(self.cost_fns)

        self.completed = False

    def get_num_bidders(self):
        return len(self.cost_fns)

    def get_quota(self):
        return self.quota

    def get_approx_cost_fns(self):
        return self.approx_cost_fns

    def get_results(self):
        return {'actual_success': self.actual_success, \
                'actual_soln': self.actual_soln, \
                'actual_cost': self.actual_cost, \
                'actual_time': self.actual_time, \

                'approx_success': self.approx_success, \
                'approx_soln': self.approx_soln, \
                'approx_cost': self.cost_w_approx_soln, \
                'approx_time': self.approx_time} if self.completed else None

    def run(self):
        if not self.completed:
            self.actual_success, self.actual_soln, self.actual_time = \
                time_method(self.determine_optimal_allocations)
            self.approx_success, self.approx_soln, self.approx_time = \
                time_method(self.determine_approx_allocations)

            objective = self.get_objective_fn(self.cost_fns)
            self.actual_cost = objective(self.actual_soln) if self.actual_success else None
            self.cost_w_approx_soln = objective(self.approx_soln) if self.approx_success else None

            self.completed = True

        return self

    def approx_cost_fns(self, cost_fns):
        return [self.approx_cost_fn(cost_fn) for cost_fn in cost_fns]

    def approx_cost_fn(self, cost_fn):
        if self.approx_method is 'least_squares':
            raw_x = np.linspace(0, cost_fn.get_capacity(), NUM_APPROX_VALS)
            y = cost_fn.eval(raw_x)
            X = np.column_stack((np.ones(raw_x.shape[0]), raw_x))
            least_squares_result = np.linalg.lstsq(X, y)
            return CostFn(least_squares_result[0], cost_fn.get_capacity())

    def determine_optimal_allocations(self):
        num_vars = len(self.cost_fns)
        objective = self.get_objective_fn(self.cost_fns)
        constraints = self.create_auction_constraints(self.cost_fns, self.quota)
        
        possible_binding_sets = it.combinations(range(1, constraints.shape[0]), num_vars - 1)
        num_vertices = 0
        min_vertex = None
        min_value = None
        for binding_set in possible_binding_sets:
            full_binding_set = (0,) + binding_set
            if Vertex.is_valid(full_binding_set, constraints):
                v = Vertex(full_binding_set, constraints)
                if v.is_feasible(constraints):
                    num_vertices += 1
                    value = objective(v.get_coords())
                    if min_value is None or value < min_value:
                        min_vertex = v
                        min_value = value

        if min_vertex is None:
            return False, np.zeros(num_vars)

        return True, min_vertex.get_coords()

    def get_objective_fn(self, cost_fns):
        def objective(x):
            # TODO: find a cleaner, more numpy-friendly way of computing objective function
            costs = np.array([pair[0].eval(pair[1]) for pair in zip(cost_fns, x.T)])
            return np.sum(costs, 0)

        return objective

    def create_auction_constraints(self, cost_fns, quota):
        num_vars = len(cost_fns)

        # from top row to bottom: quota constraint, nonnegativity constraints, capacity constraints
        A = np.concatenate((np.ones(num_vars)[np.newaxis], \
                        -1 * np.eye(num_vars), \
                        np.eye(num_vars)))

        # from top row to bottom: quota constraint, nonnegativity constraints, capacity constraints
        b = np.concatenate((np.array([quota]), \
                        np.zeros(num_vars), \
                        np.array([fn.get_capacity() for fn in cost_fns])))

        return np.column_stack((A, b))

    def get_min_vertex_from(self, vertices, objective):
        vertex_list = list(vertices)
        vertex_coords = np.array([v.get_coords() for v in vertex_list])
        return vertex_list[np.argmin(objective(vertex_coords))]

    def determine_approx_allocations(self):
        num_vars = len(self.approx_cost_fns)
        objective = self.get_linear_cost_coef(self.approx_cost_fns)
        constraints = self.create_auction_constraints(self.approx_cost_fns, self.quota)

        A_ub, b_ub = constraints[1:, :-1], constraints[1:, -1]
        A_eq, b_eq = constraints[0, :-1][np.newaxis], constraints[0, -1]
        
        result = linprog(objective, A_ub, b_ub, A_eq, b_eq)
        return result.success, result.x

    def get_linear_cost_coef(self, cost_fns):
        coef = np.array([fn.get_coef() for fn in cost_fns])
        return coef[:,1]