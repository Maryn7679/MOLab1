import matplotlib.pyplot as plt
import numpy as np
import random
import time


cases = 10
n_items = 5
size_of_bag = 15
items = [[12, 4], [1, 2], [1, 1], [2, 2], [4, 10]]


def gen_chromosomes(n):
    result = []
    for i in range(n):
        result.append(np.random.default_rng().integers(2, size=n_items))
    return result


def fitness(chromosome):
    fitness_sum = 0
    weight_sum = 0
    for gene, item in zip(chromosome, items):
        weight_sum += item[0]
        if weight_sum > size_of_bag:
            return -1
        fitness_sum += gene * item[1]
    return fitness_sum


def selection(chromosomes):
    return chromosomes[:(n_items//2)]


def crossover(chr1, chr2):
    point = random.randint(1, n_items - 1)
    return chr1[:point] + chr2[point:], chr2[:point] + chr1[point:]


def mutation(chromosome):
    i = random.randint(0, n_items)
    chromosome[i] = not chromosome[i]
    return chromosome


def chromosome_to_items(chromosome):
    result = []
    for gene, item in zip(chromosome, items):
        if gene:
            result.append(item)
    return result


def genetic_algorithm():
    fitness1 = -3
    fitness2 = -2
    fitness3 = -1
    chromosomes = gen_chromosomes(cases)
    while fitness3 > fitness1:
        chromosomes = chromosomes.sort(key=lambda x: fitness(x), reverse=True)

        fitness1 = fitness2
        fitness2 = fitness3
        fitness3 = fitness(chromosomes[0])

        chromosomes = selection(chromosomes)
        while len(chromosomes) < cases:
            a = random.randint(0, len(chromosomes))
            b = random.randint(0, len(chromosomes))
            chromosomes += crossover(chromosomes[a], chromosomes[b])

        a = random.randint(0, len(chromosomes))
        b = random.randint(0, len(chromosomes))
        chromosomes[a] = mutation(chromosomes[a])
        chromosomes[b] = mutation(chromosomes[b])
    chromosomes = chromosomes.sort(key=lambda x: fitness(x), reverse=True)
    return fitness(chromosomes[0]), sum(chromosomes[0]), chromosome_to_items(chromosomes[0])


results = genetic_algorithm()
print(f"Benefit: {results[0]}, n_items: {results[1]}, item list: {results[2]}")
# calculations2 = descent2[2]
#
# plt.plot(calculations1)
# plt.plot(calculations2)
# plt.show()
