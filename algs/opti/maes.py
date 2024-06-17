import numpy as np
from .optimizer import *


class CMAES2(Optimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy with Mirrored Sampling

    Parameters
    ----------
    func : function
        Function to optimize
    count : int
        Maximum number of generations

        (M1) Initialize(y(0), σ(0), g := 0, s(0) := 0, M(0) := I)

        Repeat
            For l := 1 To λ
                (M4) z_l(g) := N(0, I)
                (M5) d_l(g) := M(g) z_l(g)
                (M6) y_l(g) := y(g) + σ(g) d_l(g)
            End

            (M8) SortOffspringPopulation

            (M9) y(g+1) := y(g) + σ(g) <d(g)>_w
            (M10) s(g+1) := (1 - cs)s(g) + √(μeffcs(2 - cs)) <z(g)>_w

            (M11) M(g+1) := M(g)[I + c1/2 ((s(g+1)(s(g+1))^T) - I) + cmu/2 ( <z(g)(z(g))^T>_w - I )]

            (M12) σ(g+1) := σ(g) exp[ cs/cσ (||s(g+1)|| / E ||N(0, I)|| - 1) ]

            g := g + 1

            Sprawdź, czy warunki zakończenia są spełnione
            Jeśli wykryto stagnację
                Zwiększ rozmiar populacji λ
                Zainicjuj ponownie y, σ, s, M

        Until(termination condition(s) fulfilled)

    """
    def __init__(self, func, count):
        super().__init__(func, count)
        self.nn = self.f.D

        # (M1) Initialize(y(0), σ(0), g := 0, s(0) := 0, M(0) := I)
        # Initial solution
        self.xx = np.random.random(self.nn)
        # Initial mean
        self.xmean = np.copy(self.xx)
        # Initial step size
        self.sigma = 1

        # (M2) λ = 4 + int(3 * np.log(nn))
        # Population size
        self.lam = 4 + int(3 * np.log(self.nn))
        # (M2) μ = int(λ / 2)
        # Number of parents/selected points
        self.mu = int(self.lam / 2)
        # (M2) Compute weights and mueff
        # Weights for recombination
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        # Normalized weights
        self.weights = np.array([w / np.sum(self.weights) for w in self.weights])
        # Variance-effectiveness of sum w_i x_i
        self.mueff = 1 / np.sum(np.power(w, 2) for w in self.weights)

        # (M2) cs, c1, cmu parameters
        # Time constant for cumulation for sigma control
        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        # Learning rate for rank-one update
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        # Learning rate for rank-mu update
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])

        # Path for sigma
        self.ps = np.zeros(self.nn)
        # Identity matrix for the covariance matrix
        self.M = np.eye(self.nn)

    def step(self):
        # (M3) Repeat For l := 1 To λ
        # Initialize population
        pop = np.zeros((self.lam, self.nn))
        # Initialize random vectors
        z = np.zeros((self.lam, self.nn))
        # Initialize transformed vectors
        d = np.zeros((self.lam, self.nn))
        for i in range(self.lam):
            # (M4) z_l(g) := N(0, I)
            # Sample from normal distribution
            z[i] = np.random.normal(0, 1, self.nn)
            # (M5) d_l(g) := M(g) z_l(g)
            # Transform to the search space
            d[i] = np.dot(self.M, z[i])
            # (M6) y_l(g) := y(g) + σ(g) d_l(g)
            # Create new population member
            pop[i] = self.xmean + self.sigma * d[i]

        # (M8) SortOffspringPopulation
        # Evaluate fitness
        fitvals = np.zeros(self.lam)
        for i in range(self.lam):
            fitvals[i] = self.f.evaluate(pop[i])
        # Sort population by fitness
        argx = np.argsort(fitvals)
        # Check for the best solution
        if fitvals[argx[0]] < self.opti_f:
            self.opti_x = pop[argx[0]]
            self.opti_f = fitvals[argx[0]]

        # (M9) y(g+1) := y(g) + σ(g) <d(g)>_w
        # Update mean
        self.xmean = self.xmean + self.sigma * np.sum(self.weights[i] * d[argx[i]] for i in range(self.mu))

        # (M10) s(g+1) := (1 - cs)s(g) + √(μeffcs(2 - cs)) <z(g)>_w
        # Update evolution path
        zz = np.sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + c * zz

        # (M11) M(g+1) := M(g)[I + c1/2 ((s(g+1)(s(g+1))^T) - I) + cmu/2 ( <z(g)(z(g))^T>_w - I )]
        # Update covariance matrix
        one = np.eye(self.nn, self.nn)
        part1 = one
        part2o = self.ps.reshape((self.nn, 1))
        part2t = self.ps.reshape((1, self.nn))
        part2 = self.c1 / 2 * (np.dot(part2o, part2t) - one)
        part3 = np.zeros((self.nn, self.nn))
        for i in range(self.mu):
            part3o = z[argx[i]].reshape((self.nn, 1))
            part3t = z[argx[i]].reshape((1, self.nn))
            part3 += self.weights[i] * np.dot(part3o, part3t)
        part3 = self.cmu / 2 * (part3 - one)
        self.M = np.dot(self.M, part1 + part2 + part3)

        # (M12) σ(g+1) := σ(g) exp[ cs/cσ (||s(g+1)|| / E ||N(0, I)|| - 1) ]
        # Update step-size
        self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

    def run(self):
        # (M13) g := g + 1
        for i in range(self.maxgen):
            self.step()
            print(i, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x

# Until(termination condition(s) fulfilled) (M14)
