#!/usr/bin/env python
from __future__ import division, print_function

import os
import webbrowser

import cma
import cocoex
import cocopp
from numpy.random import rand

suite_name = "bbob"
output_folder = "scipy-optimize-cmaes"
optimizer = cma.fmin
budget_multiplier = 1

suite = cocoex.Suite(suite_name, "", "function_indices:1,2,3,4,5")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

for problem in suite:
    problem.observe_with(observer)
    x0 = problem.initial_solution
    while (problem.evaluations < problem.dimension * budget_multiplier and not problem.final_target_hit):
        optimizer(problem, x0, 1)
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) * (problem.upper_bounds - problem.lower_bounds) / 2)
    minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
