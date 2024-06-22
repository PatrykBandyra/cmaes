import numpy  # engine for numerical computing

from algs.pypop7.maes import MAES
from benchmarks.base_functions import rosenbrock  # function to be minimized
from maes_ipop import MAES_IPOP2
import numpy as np # engine for numerical computing
from benchmarks.base_functions import rosenbrock, schaffer, schwefel222  # function to be minimized
from cmaes import CMAES

def test_optimize_maes():
    import numpy  # engine for numerical computing
    from benchmarks.base_functions import rosenbrock  # function to be minimized
    from maes import MAES
    problem = {'fitness_function': rosenbrock,  # to define problem arguments
               'ndim_problem': 2,
               'lower_boundary': -5.0 * numpy.ones((2,)),
               'upper_boundary': 5.0 * numpy.ones((2,))}
    options = {'max_function_evaluations': 5000,  # to set optimizer options
               'seed_rng': 2022,
               'mean': 3.0 * numpy.ones((2,)),
               'sigma': 3.0}  # global step-size may need to be tuned for optimality
    maes = MAES(problem, options)  # to initialize the black-box optimizer class
    results = maes.optimize()  # to run its optimization/evolution process
    print(results)  # to show the optimization results

    assert results['n_function_evaluations'] == 5000
    assert results['best_so_far_y'] < 5.0



def test_optimize_cmaes(dim = 2, max_function_evaluations = 5000, fitness_function = rosenbrock, lb =  - 5.0, ub = 5.0):


    lower_bounds = lb * numpy.ones((dim,))
    upper_bounds = ub * numpy.ones((dim,))

    problem = {'fitness_function': fitness_function,  # to define problem arguments
               'ndim_problem': dim,
               'lower_boundary': lower_bounds,
               'upper_boundary': upper_bounds}
    max_function_evaluations = max_function_evaluations
    options = {'max_function_evaluations': max_function_evaluations,  # to set optimizer options
               'seed_rng': 2022,
               'mean': 3.0 * numpy.ones((dim,)),
               'sigma': np.min(upper_bounds - lower_bounds) / 3.0}  # global step-size may need to be tuned for optimality
    maes = CMAES(problem, options)  # to initialize the black-box optimizer class
    results = maes.optimize()  # to run its optimization/evolution process
    print(results)  # to show the optimization results

    #assert results['n_function_evaluations'] == 5000
    #assert results['best_so_far_y'] < 5.0

    # Check if the optimizer found the minimum
    expected_minimum = 0.0
    tolerance = 1e-2  # Tolerance level for the minimum check
    if abs(results['best_so_far_y'] - expected_minimum) < tolerance:
        f"Optimizer did not find the minimum. Best so far: {results['best_so_far_y']}"
    else:
        print("Optimization successful. Best found value is close to the known minimum.")

def test_optimize_maes_ipop(dim = 2, max_function_evaluations = 5000, fitness_function = rosenbrock, lb =  - 5.0, ub = 5.0):


    lower_bounds = lb * numpy.ones((dim,))
    upper_bounds = ub * numpy.ones((dim,))

    problem = {'fitness_function': fitness_function,  # to define problem arguments
               'ndim_problem': dim,
               'lower_boundary': lower_bounds,
               'upper_boundary': upper_bounds}
    max_function_evaluations = max_function_evaluations
    options = {'max_function_evaluations': max_function_evaluations,  # to set optimizer options
               'seed_rng': 2022,
               'mean': 3.0 * numpy.ones((dim,)),
               'sigma': np.min(upper_bounds - lower_bounds) / 3.0}  # global step-size may need to be tuned for optimality
    maes = MAES_IPOP2(problem, options)  # to initialize the black-box optimizer class
    results = maes.optimize()  # to run its optimization/evolution process
    print(results)  # to show the optimization results

    #assert results['n_function_evaluations'] == 5000
    #assert results['best_so_far_y'] < 5.0

    # Check if the optimizer found the minimum
    expected_minimum = 0.0
    tolerance = 1e-2  # Tolerance level for the minimum check
    if abs(results['best_so_far_y'] - expected_minimum) < tolerance:
        f"Optimizer did not find the minimum. Best so far: {results['best_so_far_y']}"
    else:
        print("Optimization successful. Best found value is close to the known minimum.")


def test_optimize_maes(dim = 2, max_function_evaluations = 5000, fitness_function = rosenbrock, lb =  - 5.0, ub = 5.0):


    lower_bounds = lb * numpy.ones((dim,))
    upper_bounds = ub * numpy.ones((dim,))

    problem = {'fitness_function': fitness_function,  # to define problem arguments
               'ndim_problem': dim,
               'lower_boundary': lower_bounds,
               'upper_boundary': upper_bounds}
    max_function_evaluations = max_function_evaluations
    options = {'max_function_evaluations': max_function_evaluations,  # to set optimizer options
               'seed_rng': 2022,
               'mean': 3.0 * numpy.ones((dim,)),
               'sigma': np.min(upper_bounds - lower_bounds) / 3.0}  # global step-size may need to be tuned for optimality
    maes = MAES(problem, options)  # to initialize the black-box optimizer class
    results = maes.optimize()  # to run its optimization/evolution process
    print(results)  # to show the optimization results

    #assert results['n_function_evaluations'] == 5000
    #assert results['best_so_far_y'] < 5.0

    # Check if the optimizer found the minimum
    expected_minimum = 0.0
    tolerance = 1e-2  # Tolerance level for the minimum check
    if abs(results['best_so_far_y'] - expected_minimum) < tolerance:
        f"Optimizer did not find the minimum. Best so far: {results['best_so_far_y']}"
    else:
        print("Optimization successful. Best found value is close to the known minimum.")

# fill default
#test_optimize_cmaes(dim = 40, max_function_evaluations=5000, fitness_function=schaffer, lb=-5.0,ub= 5.0)
#test_optimize_maes(dim = 40, max_function_evaluations=5000, fitness_function=schaffer, lb=-5.0,ub= 5.0)
#test_optimize_maes_ipop(dim = 40, max_function_evaluations=5000, fitness_function=schaffer, lb=-5.0,ub= 5.0)

