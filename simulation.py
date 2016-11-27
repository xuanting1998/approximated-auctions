import numpy as np
import scipy as sc
from auction import CostFn, Auction
import sys
import time
import multiprocessing
from functools import partial
import csv

def compute_coef(max_deg, cap, deg):
    return sc.special.binom(max_deg, deg) * (-1*cap)**(max_deg - deg)

def rand_cost_fn(deg_mean, deg_prob, cap_mean, cap_var):
    degree = np.random.binomial(deg_mean / deg_prob, deg_prob)
    capacity = np.random.normal(cap_mean, cap_var)
    coef = (-1)**(degree - 1) * np.vectorize(compute_coef, excluded={0, 1})(degree, capacity, np.arange(degree + 1))
    return CostFn(coef, capacity)

def rand_auction(deg_mean, deg_prob, cap_mean, cap_var, num_bidders, quota_factor):
    cost_fns = [rand_cost_fn(deg_mean, deg_prob, cap_mean, cap_var) for i in range(int(num_bidders))]
    total_capacity = np.sum(np.array([fn.get_capacity() for fn in cost_fns]))
    return Auction(quota_factor * total_capacity, cost_fns)

def run_simulation_with_params(params, param_names, sims_per_test, auction_params):
        auction_params_ = auction_params.copy()
        id, param_vals = params[0], params[1:]
        param_updates = {param_names[i]: param_vals[i] for i in range(len(param_names))}
        auction_params_.update(param_updates)

        auction_results = [rand_auction(**auction_params_).run().get_results() for i in range(sims_per_test)]
        percent_err = np.array([(id,) + \
                                tuple(param_vals) + \
                                (abs((r['approx_cost'] - r['actual_cost'])/r['actual_cost']),) \
                            for r in auction_results \
                            if r['actual_success'] and r['approx_success']])
        print '(id, {param_names}) = ({id}, {param_vals})'.format(param_names=param_names, id=id, param_vals=param_vals)
        return percent_err

def generate_data_with_param_vals(param_names, param_vals, sims_per_test, auction_params):
    start = time.clock()
    sim_ids = np.arange(param_vals.shape[0])
    vals_with_ids = np.column_stack((sim_ids, param_vals))
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    run_simulation = partial(run_simulation_with_params, \
                        param_names=param_names, \
                        sims_per_test=sims_per_test, \
                        auction_params=auction_params)
    raw_results = pool.map(run_simulation, vals_with_ids)
    end = time.clock()
    print 'FINISHED in {time} seconds'.format(time=(end - start))
    return [auction for sim in raw_results for auction in sim]

def generate_data_varied_by_quota(auction_params, min_quota, max_quota, num_tests, sims_per_test):
    quota_factors = np.linspace(min_quota, max_quota, num_tests)
    return generate_data_with_param_vals(('quota_factor',), quota_factors, sims_per_test, auction_params)

def generate_data_varied_by_num_bidders(auction_params, min_num_bidders, max_num_bidders, num_repeats, sims_per_test):
    auction_sizes = np.arange(min_num_bidders, max_num_bidders + 1).repeat(num_repeats)
    return generate_data_with_param_vals(('num_bidders',), auction_sizes, sims_per_test, auction_params)

def generate_data_varied_by_quota_and_bidders(auction_params, \
                                            min_num_bidders, \
                                            max_num_bidders, \
                                            min_quota, \
                                            max_quota, \
                                            num_tests, \
                                            sims_per_test):
    quota_factors = np.linspace(min_quota, max_quota, num_tests).repeat((max_num_bidders + 1 - min_num_bidders))
    auction_sizes = np.tile(np.arange(min_num_bidders, max_num_bidders + 1), num_tests)
    params = np.column_stack((quota_factors, auction_sizes))
    
    return generate_data_with_param_vals(('quota_factor', 'num_bidders'), params, sims_per_test, auction_params)

def write_data_to_csv(filename, data):
    with open(filename, 'wb') as f:
        writer = csv.writer(f)
        for entry in data:
            writer.writerow(entry)

output_suffix = sys.argv[1] # TODO: check if len(sys.argv) > 1

auction_params = {'deg_mean': 3, 'deg_prob': 0.33, 'cap_mean': 4, 'cap_var': 1, 'num_bidders': 8, 'quota_factor': 0.5}

quota_data = generate_data_varied_by_quota(auction_params, min_quota=0.1, max_quota=0.9, num_tests=50, sims_per_test=100)
bidder_count_data = generate_data_varied_by_num_bidders(auction_params, \
                                                        min_num_bidders=2, \
                                                        max_num_bidders=8, \
                                                        num_repeats=10, \
                                                        sims_per_test=100)

quota_and_bidder_data = generate_data_varied_by_quota_and_bidders(auction_params, \
                                                                    min_num_bidders=2,
                                                                    max_num_bidders=8,
                                                                    min_quota=0.1, \
                                                                    max_quota=0.9, \
                                                                    num_tests=50, \
                                                                    sims_per_test=100)

write_data_to_csv('quota_data_{suffix}.csv'.format(suffix=output_suffix), quota_data)
write_data_to_csv('bidder_count_data_{suffix}.csv'.format(suffix=output_suffix), bidder_count_data)
write_data_to_csv('quota_and_bidder_data_{suffix}.csv'.format(suffix=output_suffix), quota_and_bidder_data)