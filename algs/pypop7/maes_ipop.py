import numpy as np  # engine for numerical computing

import csv

from ES_hueristics import ES_heuristics


class MAES_IPOP(ES_heuristics):
    """Matrix Adaptation Evolution Strategy (MAES).

    .. note:: `MAES` is a powerful *simplified* version of the well-established `CMA-ES` algorithm nearly without
       significant performance loss, designed in 2017 by `Beyer <https://homepages.fhv.at/hgb/>`_ and `Sendhoff
       <https://tinyurl.com/26szwuaa>`_. One obvious advantage of such a simplification is to help better understand
       the underlying working principles (e.g., **invariance** and **unbias**) of `CMA-ES`, which are often thought to
       be rather complex for newcomers. It is **highly recommended** to first attempt more advanced ES variants (e.g.,
       `LMCMA`, `LMMAES`) for large-scale black-box optimization, since `MAES` has a *cubic* time complexity (w.r.t.
       each sampling). Note that another improved version called `FMAES` provides a *relatively more efficient*
       implementation for `MAES` with *quadratic* time complexity (w.r.t. each sampling).

    Parameters
    ----------
    problem : dict
              problem arguments with the following common settings (`keys`):
                * 'fitness_function' - objective function to be **minimized** (`func`),
                * 'ndim_problem'     - number of dimensionality (`int`),
                * 'upper_boundary'   - upper boundary of search range (`array_like`),
                * 'lower_boundary'   - lower boundary of search range (`array_like`).
    options : dict
              optimizer options with the following common settings (`keys`):
                * 'max_function_evaluations' - maximum of function evaluations (`int`, default: `np.Inf`),
                * 'max_runtime'              - maximal runtime to be allowed (`float`, default: `np.Inf`),
                * 'seed_rng'                 - seed for random number generation needed to be *explicitly* set (`int`);
              and with the following particular settings (`keys`):
                * 'sigma'         - initial global step-size, aka mutation strength (`float`),
                * 'mean'          - initial (starting) point, aka mean of Gaussian search distribution (`array_like`),

                  * if not given, it will draw a random sample from the uniform distribution whose search range is
                    bounded by `problem['lower_boundary']` and `problem['upper_boundary']`.

                * 'n_individuals' - number of offspring, aka offspring population size (`int`, default:
                  `4 + int(3*np.log(problem['ndim_problem']))`),
                * 'n_parents'     - number of parents, aka parental population size (`int`, default:
                  `int(options['n_individuals']/2)`).

    Examples
    --------
    Use the optimizer `MAES` to minimize the well-known test function
    `Rosenbrock <http://en.wikipedia.org/wiki/Rosenbrock_function>`_:

    .. code-block:: python
       :linenos:

       >>> import numpy  # engine for numerical computing
       >>> from pypop7.benchmarks.base_functions import rosenbrock  # function to be minimized
       >>> from pypop7.optimizers.es.maes import MAES
       >>> problem = {'fitness_function': rosenbrock,  # define problem arguments
       ...            'ndim_problem': 2,
       ...            'lower_boundary': -5.0*numpy.ones((2,)),
       ...            'upper_boundary': 5.0*numpy.ones((2,))}
       >>> options = {'max_function_evaluations': 5000,  # set optimizer options
       ...            'seed_rng': 2022,
       ...            'mean': 3.0*numpy.ones((2,)),
       ...            'sigma': 0.1}  # the global step-size may need to be tuned for better performance
       >>> maes = MAES(problem, options)  # initialize the optimizer class
       >>> results = maes.optimize()  # run the optimization process
       >>> # return the number of function evaluations and best-so-far fitness
       >>> print(f"MAES: {results['n_function_evaluations']}, {results['best_so_far_y']}")
       MAES: 5000, 2.129367016460251e-19

    For its correctness checking of coding, refer to `this code-based repeatability report
    <https://tinyurl.com/3zvve79b>`_ for more details.

    Attributes
    ----------
    mean          : `array_like`
                    initial (starting) point, aka mean of Gaussian search distribution.
    n_individuals : `int`
                    number of offspring, aka offspring population size.
    n_parents     : `int`
                    number of parents, aka parental population size.
    sigma         : `float`
                    final global step-size, aka mutation strength.

    References
    ----------
    Beyer, H.G., 2020, July.
    `Design principles for matrix adaptation evolution strategies.
    <https://dl.acm.org/doi/abs/10.1145/3377929.3389870>`_
    In Proceedings of ACM Conference on Genetic and Evolutionary Computation Companion (pp. 682-700).

    Loshchilov, I., Glasmachers, T. and Beyer, H.G., 2019.
    `Large scale black-box optimization by limited-memory matrix adaptation.
    <https://ieeexplore.ieee.org/abstract/document/8410043>`_
    IEEE Transactions on Evolutionary Computation, 23(2), pp.353-358.

    Beyer, H.G. and Sendhoff, B., 2017.
    `Simplify your covariance matrix adaptation evolution strategy.
    <https://ieeexplore.ieee.org/document/7875115>`_
    IEEE Transactions on Evolutionary Computation, 21(5), pp.746-759.

    See the official Matlab version from Prof. Beyer:
    https://homepages.fhv.at/hgb/downloads/ForDistributionFastMAES.tar
    """

    def __init__(self, problem, options):
        ES_heuristics.__init__(self, problem, options)
        self.fitness_function = problem['fitness_function']
        # File to store stagnation reasons and ipop
        self.ipop_file = "ipop_log.csv"

        self.options = options
        self.c_s = None
        self.c_1 = None
        self.c_w = None
        self.d_sigma = None
        self._alpha_cov = 2.0
        self.max_populationsize = 5000
        self._s_1 = None
        self._s_2 = None
        self._fast_version = options.get('_fast_version', False)
        if not self._fast_version:
            self._diag_one = np.eye(self.ndim_problem)

        # Dynamic population size based on lambda calculation
        #self.n_individuals = options.get('n_individuals', 4 + int(3 * np.log(problem['ndim_problem'])))

        # if self.n_individuals > self.max_populationsize:
        #     self.n_individuals = self.max_populationsize

        self._initialize_history()

    def _initialize_history(self):
        self.history = []

    def _set_c_w(self):
        return np.minimum(1.0 - self.c_1, self._alpha_cov * (self._mu_eff + 1.0 / self._mu_eff - 2.0) /
                          (np.power(self.ndim_problem + 2.0, 2) + self._alpha_cov * self._mu_eff / 2.0))

    def _set_d_sigma(self):
        return 1.0 + self.c_s + 2.0 * np.maximum(0.0, np.sqrt((self._mu_eff - 1.0) / (self.ndim_problem + 1.0)) - 1.0)

    def initialize(self, is_restart=False):
        self.history = []
        self.c_s = self.options.get('c_s', (self._mu_eff + 2.0) / (self._mu_eff + self.ndim_problem + 5.0))
        self.c_1 = self.options.get('c_1', self._alpha_cov / (np.power(self.ndim_problem + 1.3, 2) + self._mu_eff))
        self.c_w = self.options.get('c_w', self._set_c_w())
        self.d_sigma = self.options.get('d_sigma', self._set_d_sigma())
        self._s_1 = 1.0 - self.c_s
        self._s_2 = np.sqrt(self._mu_eff * self.c_s * (2.0 - self.c_s))
        z = np.empty((self.n_individuals, self.ndim_problem))  # Gaussian noise for mutation
        d = np.empty((self.n_individuals, self.ndim_problem))  # search directions
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution
        s = np.zeros((self.ndim_problem,))  # evolution path
        tm = np.eye(self.ndim_problem)  # transformation matrix
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        self._list_initial_mean.append(np.copy(mean))
        self._initialize_history()
        return z, d, mean, s, tm, y

    def iterate(self, z=None, d=None, mean=None, tm=None, y=None, args=None):
        for k in range(self.n_individuals):  # to sample offspring population
            if self._check_terminations():
                return z, d, y
            z[k] = self.rng_optimization.standard_normal((self.ndim_problem,))
            d[k] = np.dot(tm, z[k])
            y[k] = self._evaluate_fitness(mean + self.sigma * d[k], args)
        return z, d, y

    def _update_distribution(self, z=None, d=None, mean=None, s=None, tm=None, y=None):
        order = np.argsort(y)
        d_w, z_w, zz_w = np.zeros((self.ndim_problem,)), np.zeros((self.ndim_problem,)), None
        if not self._fast_version:
            zz_w = np.zeros((self.ndim_problem, self.ndim_problem))
        for k in range(self.n_parents):
            d_w += self._w[k] * d[order[k]]
            z_w += self._w[k] * z[order[k]]
            if not self._fast_version:
                zz_w += self._w[k] * np.outer(z[order[k]], z[order[k]])
        # update distribution mean
        mean += self.sigma * d_w
        # update evolution path (s) and transformation matrix (M)
        s = self._s_1 * s + self._s_2 * z_w
        if not self._fast_version:
            tm_1 = self.c_1 * (np.outer(s, s) - self._diag_one)
            tm_2 = self.c_w * (zz_w - self._diag_one)
            tm += 0.5 * np.dot(tm, tm_1 + tm_2)
        else:
            tm = (1.0 - 0.5 * (self.c_1 + self.c_w)) * tm + (0.5 * self.c_1) * np.dot(
                np.dot(tm, s[:, np.newaxis]), s[np.newaxis, :])
            for k in range(self.n_parents):
                tm += (0.5 * self.c_w) * self._w[k] * np.outer(d[order[k]], z[order[k]])
        # update global step-size
        self.sigma *= np.exp(self.c_s / self.d_sigma * (np.linalg.norm(s) / self._e_chi - 1.0))
        return mean, s, tm

    def restart_reinitialize(self, z=None, d=None, mean=None, s=None, tm=None, y=None):
        if ES_heuristics.restart_reinitialize(self, y):
            z, d, mean, s, tm, y = self.initialize(True)
        return z, d, mean, s, tm, y

    def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
        fitness = ES_heuristics.optimize(self, fitness_function)
        z, d, mean, s, tm, y = self.initialize()
        while not self.termination_signal:
            # sample and evaluate offspring population
            z, d, y = self.iterate(z, d, mean, tm, y, args)
            if self._check_terminations():
                break
            mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
            self._print_verbose_info(fitness, y)
            self._n_generations += 1

            if self.check_stagnation(y, s, tm, mean):
                z, d, mean, s, tm, y = self.restart_reinitialize(z, d, mean, s, tm, y)
        results = self._collect(fitness, y, mean)
        results['s'] = s
        # by default, do NOT save transformation matrix of search distribution in order to save memory,
        # owing to its *quadratic* space complexity
        return results

    def log_stagnation_reasons(self, reasons):
        with open(self.ipop_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            individuals = self.n_individuals
            writer.writerow([self._n_generations, self.fitness_function, self.ndim_problem, individuals, ', '.join(reasons)])


    def increment_pop_size(self):
        print("Population size: ", self.n_individuals)
        self.n_individuals *= 2  # Doubling the population size on stagnation
        print("Increased population size to: ", self.n_individuals)

    def check_stagnation(self, y, s, tm, mean):
        reasons = []

        # heuristic 0:
        min_y = np.min(y)
        if min_y < self._list_fitness[-1]:
            self._list_fitness.append(min_y)
        else:
            self._list_fitness.append(self._list_fitness[-1])
        is_restart_1, is_restart_2 = self.sigma < self.sigma_threshold, False
        if len(self._list_fitness) >= self.stagnation:
            is_restart_2 = (self._list_fitness[-self.stagnation] - self._list_fitness[-1]) < self.fitness_diff
        if bool(is_restart_1) or bool(is_restart_2):
            reasons.append("Stagnation detected.")

        # Heuristic 1: All fitness values are NaN or infinite
        if np.all(np.isnan(y)) or np.all(np.isinf(y)):
            reasons.append("All fitness values are NaN or infinite.")

        # Heuristic 2: Covariance matrix values too high
        if np.linalg.cond(tm) > 1e14:
            reasons.append(f"Covariance matrix condition number is too high: {np.linalg.cond(tm)}")

        if self.history is None:
            self.history = []

        # Heuristic 3: No progress in optimization
        self.history.append(y[0])
        if len(self.history) > (10.0 + ((30.0 * self.ndim_problem) / self.n_individuals)):
            self.history.pop(0)

        tol_fitness_function = 1e-12

        # sprawdzamy czy historia jest dostatecznie długa aby mierzyc stagnacje
        minimum_history_length = (len(self.history) == int(10.0 + ((30.0 * self.ndim_problem) / self.n_individuals)))

        fitness_min_equals_max = (np.max(self.history) == np.min(self.history))
        fitness_min_almost_eq_max = (np.max(self.history) - np.min(self.history)) < tol_fitness_function
        fitness_function_peek2peek_too_small = (np.ptp(y) < tol_fitness_function)

        fitness_fun_all_conditions = \
            (fitness_min_equals_max or (fitness_min_almost_eq_max and fitness_function_peek2peek_too_small))

        is_objective_function_range_too_low = minimum_history_length and fitness_fun_all_conditions

        if is_objective_function_range_too_low:
            reasons.append("No progress in optimization: range of best objective function values is too low.")

        # Heuristic 4: Standard deviation too small
        tolX = 1e-12 * self.sigma
        all_els_evolution_path_too_small = np.all(np.abs(s) < tolX)
        all_matrix_m_elements_too_small = np.all(np.abs(tm) < tolX)
        is_standard_deviation_too_small = (all_matrix_m_elements_too_small and all_els_evolution_path_too_small)

        if is_standard_deviation_too_small:
            reasons.append(f"Standard deviation too small: tolX = {tolX}")

        # Heuristic 5: No effect axis
        ith = (self._n_generations % self.ndim_problem)
        tmpXmean = mean + 0.1 * self.sigma * np.dot(tm, np.random.normal(size=self.ndim_problem))

        is_no_effect_axis = (ith < len(tmpXmean) and tmpXmean[ith] == mean[ith])

        if is_no_effect_axis:
            reasons.append(
                "No effect axis: adding 0.1-standard deviation vector in a principal axis direction does not change the mean.")

        # Heuristic 6: No effect coordinate
        testC = 0.2 * self.sigma * tm
        eigvals, eigvecs = np.linalg.eigh(testC)
        col_values = np.sqrt(np.abs(eigvals))
        tmpXmean = mean + self.sigma * np.dot(eigvecs, col_values)

        is_no_effect_coord = np.all(mean == tmpXmean)

        if is_no_effect_coord:
            reasons.append(
                "No effect coordinate: adding 0.2-standard deviation in each coordinate does not change the mean.")

        if reasons:
            print(
                f"Stagnation detected in iteration {self._n_generations}, function {self.fitness_function} and dimensions {self.ndim_problem} for the following reasons:")
            for reason in reasons:
                print(reason)
            self.log_stagnation_reasons(reasons)
            return True

        return False