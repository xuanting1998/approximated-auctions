import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.optimize import minimize

NUM_APPROX_VALS = 200

class CostFn(object):
    def __init__(self, coefficients, capacity):
        self.__fn = P(coefficients)
        self.__capacity = capacity

    def eval(self, values):
        return self.__fn(values)

    def get_capacity(self):
        return self.__capacity

    def get_derivative(self):
        return self.__fn.deriv(1)

    def get_coef(self):
        return self.__fn.coef

class Auction(object):
    def __init__(self, quota, cost_fns, approx_method = 'least_squares'):
        self.__quota = quota
        self.__approx_method = approx_method

        self.__cost_fns = cost_fns
        self.__approx_cost_fns = self.__approx_cost_fns(self.__cost_fns)

        self.__completed = False

    def get_num_bidders(self):
        return len(self.__cost_fns)

    def get_quota(self):
        return self.__quota

    def get_approx_cost_fns(self):
        return self.__approx_cost_fns

    def get_results(self):
        return {'actual_soln': self.__actual_min_soln, \
                'actual_cost': self.__actual_min_cost, \
                'approx_soln': self.__approx_min_soln, \
                'approx_cost': self.__cost_w_approx_soln} if self.__completed else None

    def run(self):
        if not self.__completed:
            initial_guess = np.zeros(self.get_num_bidders())
            quota_constraint = ({ \
                'type': 'eq', \
                'fun': lambda x: np.sum(x) - self.get_quota(), \
                'jac': lambda x: np.ones(self.get_num_bidders())
            })
            capacity_constraints = [(0, fn.get_capacity()) for fn in self.__cost_fns]

            objective, deriv = self.__get_objective_fn(self.__cost_fns)

            actual_min_results = minimize(objective, \
                            initial_guess, \
                            args=(), \
                            jac=deriv, \
                            constraints=quota_constraint, \
                            bounds=capacity_constraints)

            approx_objective, approx_deriv = self.__get_objective_fn(self.__approx_cost_fns)

            approx_min_results = minimize(approx_objective, \
                            initial_guess, \
                            args=(), \
                            jac=approx_deriv, \
                            constraints=quota_constraint, \
                            bounds=capacity_constraints)

            self.__actual_min_soln = actual_min_results.x
            self.__actual_min_cost = actual_min_results.fun
            self.__approx_min_soln = approx_min_results.x
            self.__cost_w_approx_soln = objective(self.__actual_min_soln)

            self.__completed = True

    def __approx_cost_fns(self, cost_fns):
        return [self.__approx_cost_fn(cost_fn) for cost_fn in cost_fns]

    def __approx_cost_fn(self, cost_fn):
        if self.__approx_method is 'least_squares':
            raw_x = np.linspace(0, cost_fn.get_capacity(), NUM_APPROX_VALS)
            y = cost_fn.eval(raw_x)
            X = np.column_stack((np.ones(raw_x.shape[0]), raw_x))
            least_squares_result = np.linalg.lstsq(X, y)
            return CostFn(least_squares_result[0], cost_fn.get_capacity())

    def __get_objective_fn(self, cost_fns):
        def objective(x):
            # TODO: find a cleaner, more numpy-friendly way of computing objective function
            costs = np.array([pair[0].eval(pair[1]) for pair in zip(cost_fns, x.T)])
            return np.sum(costs, 0)

        deriv_fns = [fn.get_derivative() for fn in cost_fns]

        def deriv(x):
            # TODO: find a cleaner, more numpy-friendly way of computing objective function
            return np.array([pair[0](pair[1]) for pair in zip(deriv_fns, x.T)])

        return objective, deriv