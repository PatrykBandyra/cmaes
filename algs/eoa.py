from prob.problems import *
from opti.de import DE

from opti.cmaes import CMAES
from opti.cmaes_origin import CMAESO
from opti.cmaes_maes import CMAESM
from opti.cmaes_large import CMAESL

# beta
from opti.cmaes_bipop import CMAESB

if __name__ == "__main__":
    TaskProb = Rosenbrock(2, -50, 50)
    Task = CMAES(TaskProb, 5000)
    Task.run()
