""" 
    Universidad del Valle de Guatemala. Guatemala, noviembre 2020.
    @autores Ana Lucía Hernández y María Fernanda López. 

    Funciones que constituye el funcionamiento del algoritmo genético. 
"""

import numpy as np
from itertools import combinations

def generate_population(n, min_lim, max_lim, evaluate):
    population = []
    # se generan tantos individuos como lo indique n
    for p in range(n):
        p1 = np.random.randint(min_lim, max_lim)
        p2 = np.random.randint(min_lim, max_lim)
        if evaluate(p1, p2):
            population.append(np.array([p1, p2]))
    return population

def get_fit_end(population, fit_function):
    fit_score = [
        fit_function(x[0], x[1]) for x in population
    ]
    fit_score.sort(reverse = True)
    return fit_score[0]

def genetic_algorithm(rounds, population, evaluation_f, evaluate, max_evaluate):
    count = 0
    while count < rounds:
        fit_score = [
            evaluation_f(x[0], x[1]) for x in population
        ]
        score = fit_score.copy()
        fit_score.sort(reverse = True)
        
        if fit_score[0] > max_evaluate:
            max_evaluate = fit_score[0]
        elif fit_score[0] == max_evaluate:
            count += 1

        # seleccionar individuos para reproducirse
        # seleccion de genotipos
        gen1 = score.index(fit_score[0])
        gen2 = score.index(fit_score[1])
        # cross-over de padres y se genera un nuevo individuo
        parent1 = population[gen1]
        parent2 = population[gen2]
        child = combinations(np.concatenate((parent1, parent2)), 2)
        population = [] # siguiente nivel de arbol genealogico
        population.append(parent1)
        population.append(parent2)

        # mutacion aleatoria de siguiente generacion
        for gen in list(child):
            child_genome = np.array(gen)
            mutation_prob = np.random.uniform() # probabilidad que el individuo tenga una mutacion
            mutated_chromosome = np.random.randint(0,2)
            if mutation_prob > 0.8:
                child_genome[mutated_chromosome] += 1
            if evaluate(child_genome[0], child_genome[1]):
                population.append(child_genome)
    return population[0], max_evaluate

"""
    Definición de funciones de cada uno de los tasks y sus respectivas condiciones.
"""
def task1(x1, x2):
    return 15*x1 + 30*x2 + 4*x1*x2 - 2*x1**2 - 4*x2**2

def task2(x1, x2):
    return 3*x1 + 5*x2

def task3(x1, x2):
    return 5*x1 - x1**2 + 8*x2 - 2*x2**2

def eval1(x1, x2):
    if (x1 + 2*x2) <= 30:
        return True
    else:
        return False

def eval2(x1, x2):
    if (3*x1 + 2*x2) <= 18:
        return True
    else:
        return False

def eval3(x1, x2):
    if (3*x1 + 2*x2) <= 6:
        return True
    else:
        return False