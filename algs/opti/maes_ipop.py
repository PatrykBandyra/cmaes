import numpy as np
from .optimizer import *


class CMAESM_IPOP(Optimizer):
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

    def __init__(self, func, count, maxgen, tol=1e-12, cov_tol=1e14):
        super().__init__(func, count)
        self.nn = self.f.D
        self.maxgen = maxgen
        self.tol = tol
        self.cov_tol = cov_tol

        # Inicjalizacja rozmiaru populacji
        self.init_population_size()

    def init_population_size(self):
        # (M1) Initialize(y(0), σ(0), g := 0, s(0) := 0, M(0) := I)
        # Początkowe rozwiązanie
        self.xx = np.random.random(self.nn)
        # Początkowa średnia
        self.xmean = np.copy(self.xx)
        # Początkowy rozmiar kroku
        self.sigma = 1

        # (M2) λ = 4 + int(3 * np.log(nn))
        # Rozmiar populacji
        self.lam = 4 + int(3 * np.log(self.nn))
        # (M2) μ = int(λ / 2)
        # Liczba rodziców/wybranych punktów
        self.mu = int(self.lam / 2)
        # (M2) Obliczanie wag i mueff
        # Wagi do rekombinacji
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        # Normalizacja wag
        self.weights = np.array([w / np.sum(self.weights) for w in self.weights])
        # Wariancja efektywna sumy w_i x_i
        self.mueff = 1 / np.sum(np.power(w, 2) for w in self.weights)

        # (M2) Parametry cs, c1, cmu
        # Stała czasowa dla kumulacji do kontroli sigma
        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        # Współczynnik uczenia dla aktualizacji rzędu jeden
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        # Współczynnik uczenia dla aktualizacji rzędu mu
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])

        # Ścieżka dla sigma
        self.ps = np.zeros(self.nn)
        # Macierz jednostkowa dla macierzy kowariancji
        self.M = np.eye(self.nn)

    def check_stagnation(self, fitvals, generation):
        """
        Heurystyki do wykrywania stagnacji
        """
        # Heurystyka 1: Wszystkie wartości funkcji celu są NaN lub nieskończone
        if np.all(np.isnan(fitvals)) or np.all(np.isinf(fitvals)):
            return True

        # Heurystyka 2: Zbyt wysokie wartości macierzy kowariancji
        if np.any(np.abs(self.M) > self.cov_tol):
            return True

        # Heurystyka 3: Brak postępu w optymalizacji
        if generation > 10:
            recent_fits = fitvals[-10:]
            if np.ptp(recent_fits) < self.tol:
                return True

        # Heurystyka 4: Zbyt małe odchylenie standardowe
        if np.all(self.sigma * np.sqrt(np.diag(self.M)) < self.tol):
            return True

        # Heurystyka 5: Dodanie wektora odchyleń standardowych w kierunku głównej osi nie zmienia średniej
        if np.allclose(self.xmean + self.sigma * self.ps, self.xmean, atol=self.tol):
            return True

        # Heurystyka 6: Dodanie ułamka odchylenia standardowego w każdej współrzędnej nie zmienia średniej
        if np.allclose(self.xmean + self.sigma * np.sqrt(np.diag(self.M)) / 2, self.xmean, atol=self.tol):
            return True

        return False

    def step(self):
        # (M3) Powtarzaj dla l := 1 do λ
        # Inicjalizacja populacji
        pop = np.zeros((self.lam, self.nn))
        # Inicjalizacja wektorów losowych
        z = np.zeros((self.lam, self.nn))
        # Inicjalizacja przekształconych wektorów
        d = np.zeros((self.lam, self.nn))
        for i in range(self.lam):
            # (M4) z_l(g) := N(0, I)
            # Próba z rozkładu normalnego
            z[i] = np.random.normal(0, 1, self.nn)
            # (M5) d_l(g) := M(g) z_l(g)
            # Przekształcenie do przestrzeni poszukiwań
            d[i] = np.dot(self.M, z[i])
            # (M6) y_l(g) := y(g) + σ(g) d_l(g)
            # Utworzenie nowego członka populacji
            pop[i] = self.xmean + self.sigma * d[i]

        # (M8) Sortowanie populacji potomnej
        # Ocena dopasowania
        fitvals = np.zeros(self.lam)
        for i in range(self.lam):
            fitvals[i] = self.f.evaluate(pop[i])
        # Sortowanie populacji według dopasowania
        argx = np.argsort(fitvals)
        # Sprawdzenie najlepszego rozwiązania
        if fitvals[argx[0]] < self.opti_f:
            self.opti_x = pop[argx[0]]
            self.opti_f = fitvals[argx[0]]

        # (M9) y(g+1) := y(g) + σ(g) <d(g)>_w
        # Aktualizacja średniej
        self.xmean = self.xmean + self.sigma * np.sum(self.weights[i] * d[argx[i]] for i in range(self.mu))

        # (M10) s(g+1) := (1 - cs)s(g) + √(μeffcs(2 - cs)) <z(g)>_w
        # Aktualizacja ścieżki ewolucji
        zz = np.sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + c * zz

        # (M11) M(g+1) := M(g)[I + c1/2 ((s(g+1)(s(g+1))^T) - I) + cmu/2 ( <z(g)(z(g))^T>_w - I )]
        # Aktualizacja macierzy kowariancji
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
        # Aktualizacja rozmiaru kroku
        self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

        return fitvals

    def run(self):
        generation = 0
        while generation < self.maxgen:
            fitvals = self.step()
            generation += 1
            if self.check_stagnation(fitvals, generation):
                print(f"Stagnation detected at generation {generation}. Restarting with increased population size.")
                self.lam = int(self.lam * 1.5)
                self.init_population_size()
            print(generation, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x

# Usage example
# optimizer = CMAESM_IPOP(your_function, count, maxgen)
# optimizer.run()
# opti_f, opti_x = optimizer.output()