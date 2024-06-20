def test_optimize_maes_ipop():
    import numpy  # engine for numerical computing
    from benchmarks.base_functions import rosenbrock  # function to be minimized
    from maes_ipop import MAES_IPOP
    problem = {'fitness_function': rosenbrock,  # to define problem arguments
               'ndim_problem': 2,
               'lower_boundary': -5.0 * numpy.ones((2,)),
               'upper_boundary': 5.0 * numpy.ones((2,))}
    options = {'max_function_evaluations': 50000,  # to set optimizer options
               'seed_rng': 2022,
               'mean': 3.0 * numpy.ones((2,)),
               'sigma': 3.0}  # global step-size may need to be tuned for optimality
    maes = MAES_IPOP(problem, options)  # to initialize the black-box optimizer class
    results = maes.optimize()  # to run its optimization/evolution process
    print(results)  # to show the optimization results

    #assert results['n_function_evaluations'] == 5000
    assert results['best_so_far_y'] < 0.01

test_optimize_maes_ipop()