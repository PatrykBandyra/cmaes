def __init__(self, problem, options):
    ES.__init__(self, problem, options)
    self.options = options
    self.c_s = None
    self.c_1 = None
    self.c_w = None
    self.d_sigma = None
    self._alpha_cov = 2.0
    self._s_1 = None
    self._s_2 = None
    self._fast_version = options.get('_fast_version', False)
    if not self._fast_version:
        self._diag_one = np.eye(self.ndim_problem)


def _set_c_w(self):
    return np.minimum(1.0 - self.c_1, self._alpha_cov * (self._mu_eff + 1.0 / self._mu_eff - 2.0) /
                      (np.power(self.ndim_problem + 2.0, 2) + self._alpha_cov * self._mu_eff / 2.0))


def _set_d_sigma(self):
    return 1.0 + self.c_s + 2.0 * np.maximum(0.0, np.sqrt((self._mu_eff - 1.0) / (self.ndim_problem + 1.0)) - 1.0)


def initialize(self, is_restart=False):
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
    if ES.restart_reinitialize(self, y):
        z, d, mean, s, tm, y = self.initialize(True)
    return z, d, mean, s, tm, y


def optimize(self, fitness_function=None, args=None):  # for all generations (iterations)
    fitness = ES.optimize(self, fitness_function)
    z, d, mean, s, tm, y = self.initialize()
    while not self.termination_signal:
        # sample and evaluate offspring population
        z, d, y = self.iterate(z, d, mean, tm, y, args)
        if self._check_terminations():
            break
        mean, s, tm = self._update_distribution(z, d, mean, s, tm, y)
        self._print_verbose_info(fitness, y)
        self._n_generations += 1
        if self.is_restart:
            z, d, mean, s, tm, y = self.restart_reinitialize(z, d, mean, s, tm, y)
    results = self._collect(fitness, y, mean)
    results['s'] = s
    # by default, do NOT save transformation matrix of search distribution in order to save memory,
    # owing to its *quadratic* space complexity
    return results