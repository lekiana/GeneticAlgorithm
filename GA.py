import random
import readData
import xlsxwriter as excel_writer
from time import perf_counter
from itertools import combinations
import sys
import numpy as np

global matrix, dimension, pool  # current path pool, updated every generation
population_size = 50
generations = 500  # stop condition - number of iterations
crossover_p = 0.9
mutation_p = 0.01
allowed_rep = 20  # stop condition - number of iterations without improving the solution
t_max = 60  # stop condition - time [s]


class Specimen:
    def __init__(self, path, cost):
        self.path = path
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __gt__(self, other):
        return self.cost > other.cost


class Population:
    def __init__(self, size):
        self.size = size
        self.specimens = list()  # object pool


def random_init(self):
    while len(self.specimens) < self.size:
        spec_path = list(range(1, dimension))
        random.shuffle(spec_path)
        spec_path = [0] + list(spec_path)
        specimen = Specimen(spec_path, get_cost(spec_path))
        if spec_path not in pool:
            pool.append(spec_path)
            self.specimens.append(specimen)
    print('best ini: ', find_best(self))
    return self


def find_closest(path):
    city_cur = path[-1]  # last city on current path
    city_best = 0
    best_dist = sys.float_info.max
    for i in range(len(matrix)):
        if i not in path and i != city_cur:
            dist = matrix[city_cur][i]
            if dist < best_dist:
                city_best = i
                best_dist = dist
    path.append(city_best)
    return path


def greedy_init(self):
    while len(self.specimens) < self.size:
        path = list()
        path.append(0)
        path.append(random.randint(1, dimension-1))
        for j in range(len(matrix)):
            path = find_closest(path)
        specimen = Specimen(path, get_cost(path))
        if path not in pool:  # to keep the paths from repeating
            pool.append(path)
            self.specimens.append(specimen)
    print('best ini: ', find_best(self))
    return self


def get_cost(path):
    dist = 0
    for i in range(dimension-2):
        dist += matrix[path[i]][path[i + 1]]
    dist += matrix[path[0]][path[dimension - 2]]
    return dist


def choose_parents(population):  # tournament selection
    # number of parents - half the number of population
    parents = set()
    it = round(population.size/2)
    while len(parents) < it:
        while True:
            participant1 = random.choice(list(population.specimens))
            participant2 = random.choice(list(population.specimens))
            if participant1 is not participant2:
                break
        if participant1 < participant2:
            if participant1 in parents:
                continue
            parents.add(participant1)
        else:
            if participant2 in parents:
                continue
            parents.add(participant2)
    return list(parents)


def choose_parents2(population):  # ranking selection
    current_pop = population.specimens
    parents = list()
    current_pop.sort(reverse=False, key=sort_funct)
    it = round(population.size/2)
    for i in range(it):
        parents.append(current_pop[i])
    return parents


def crossover_logic(parent1, parent2, idx_start, idx_end):
    fragment1 = []
    fragment2 = parent1[idx_start:idx_end]
    fragment3 = []

    copied_set = set(fragment2)

    for vertex in parent2:
        if vertex not in copied_set:
            if len(fragment1) < idx_start:
                fragment1.append(vertex)
            else:
                fragment3.append(vertex)

    path = fragment1 + fragment2 + fragment3
    child = Specimen(path, get_cost(path))

    return path, child


def crossover(parents):  # Order Crossover
    comb = list(combinations(parents, 2))
    children = []
    for pair in comb:
        if random.random() <= crossover_p:
            parent1 = pair[0].path
            parent2 = pair[1].path

            min_len = 5  # min length of the copied fragment
            max_len = 10  # max length of the copied fragment
            idx_start = random.randint(1, dimension - max_len)
            cross_over_size = random.randint(min_len, max_len)
            idx_end = idx_start + cross_over_size

            cl = crossover_logic(parent1, parent2, idx_start, idx_end)
            if cl[0] not in pool:
                pool.append(cl[0])
                children.append(cl[1])

            cl = crossover_logic(parent2, parent1, idx_start, idx_end)
            if cl[0] not in pool:
                pool.append(cl[0])
                children.append(cl[1])

    return children


def crossover2(parents):  # Random operator with completion - not used in the study (long execution time)
    comb = list(combinations(parents, 2))
    children = []
    for pair in comb:
        if random.random() <= 0.8:
            parent1 = pair[0].path
            parent2 = pair[1].path

            cities = list()
            child_path1 = list()
            child_path2 = list()
            for i in range(1, dimension):  # create an empty child path
                cities.append(i)
                child_path1.append(-1)
                child_path2.append(-1)

            indices = random.sample(cities, round(dimension / 2))  # select unchanged vertices from parent1
            indices.append(0)
            for i in range(dimension-1):
                if i in indices:
                    child_path1[i] = parent1[i]
                    child_path2[i] = parent2[i]

            for i in parent2:  # look for unused vertices in parent2 and empty fields in the child path
                for j in range(len(child_path1)):
                    if i not in child_path1:
                        if child_path1[j] == -1:  # long execution
                            child_path1[j] = i

            for i in parent1:
                for j in range(len(child_path2)):
                    if i not in child_path2:
                        if child_path2[j] == -1:  # long execution
                            child_path2[j] = i

            child1 = Specimen(child_path1, get_cost(child_path1))
            child2 = Specimen(child_path2, get_cost(child_path2))

            if child_path1 not in pool:
                pool.append(child_path1)
                children.append(child1)

            if child_path2 not in pool:
                pool.append(child_path2)
                children.append(child2)
    return children


def mutate_inversion(specimen_pool):
    for specimen in specimen_pool:
        if random.random() <= mutation_p:
            path = specimen.path
            while True:  # arc swap (inversion mutation)
                v1 = random.randint(1, len(path))
                v2 = random.randint(1, len(path))
                if v1 != v2:
                    break
            v1, v2 = sorted([v1, v2])
            bow = path[v1:v2]
            bow.reverse()
            path[v1:v2] = bow
            specimen.path = path
            specimen.cost = get_cost(path)


def mutate_swap_2(specimen_pool):
    for specimen in specimen_pool:
        if random.random() <= mutation_p:
            path = specimen.path
            while True:  # random selection of two cities (indices)
                v1 = int(np.floor(np.random.uniform(1, len(path))))
                v2 = int(np.floor(np.random.uniform(1, len(path))))
                if v1 != v2:
                    break
            path[v1], path[v2] = path[v2], path[v1]
            specimen.path = path
            specimen.cost = get_cost(path)


def sort_funct(s):
    return s.cost


def new_population(specimen_pool):
    global pool
    specimen_pool.sort(reverse=False, key=sort_funct)
    while len(specimen_pool) > population_size:  # pop_size -> end
        specimen_pool.pop()
    new_pop = Population(population_size)
    new_pop.specimens = specimen_pool
    pool = list()
    for spec in specimen_pool:
        pool.append(spec.path)
    return new_pop


def find_best(population):
    specimens = population.specimens
    specimens.sort(reverse=False, key=sort_funct)
    return specimens[0].cost


def genetic_algorithm(opt, start_time, er):
    global pool
    pool = list()

    population = Population(population_size)
    gi = greedy_init(population)  # initialize x initial paths with cost (object attributes)
    best_found = find_best(gi)
    error = abs(best_found - opt) / opt * 100

    rep_count = 0
    prev_cost = sys.maxsize

    for i in range(generations):

        cur_time = perf_counter()
        elapsed = cur_time - start_time
        if elapsed >= t_max:
            print('t > t_max')
            break

        parents = choose_parents2(population)
        children = crossover(parents)

        specimen_pool = list()
        for spec in population.specimens:  # add parents
            specimen_pool.append(spec)
        for child in children:
            specimen_pool.append(child)

        costs = list()

        mutate_inversion(specimen_pool)

        population = new_population(specimen_pool)  # specimen_pool / children

        for specimen in population.specimens:
            costs.append(get_cost(specimen.path))

        if costs[0] == prev_cost:
            rep_count += 1
        if rep_count == allowed_rep:
            print("No solution improvement since " + str(allowed_rep) + " iterations")
            break

        prev_cost = costs[0]
        pool.clear()

        best_found = find_best(population)
        error = abs(best_found - opt) / opt * 100
        if error <= er:
            print("A satisfactory solution has been found")
            break

    return best_found, error


def excel_write(wb, it, opt, file):
    worksheet = wb.add_worksheet(file)
    worksheet.write(0, 0, 'Instance name')
    worksheet.write(1, 0, file)
    worksheet.write(0, 2, 'Iterations')
    worksheet.write(1, 2, it)
    worksheet.write(0, 4, 'Optimal value')
    worksheet.write(1, 4, opt)
    worksheet.write(3, 0, 'Execution time [s]: ')
    worksheet.write(3, 2, 'Cost obtained: ')
    worksheet.write(3, 4, 'Error [%]: ')
    return worksheet


def main():
    # READ CONFIGURATION FILE #
    config_data = readData.read_config()
    data_files = config_data[0]
    iterators = config_data[1]
    optimal_values = config_data[2]
    output_files = config_data[4]

    # CREATE OUTPUT FILE(S)#
    workbooks = list()
    for file in output_files:
        workbooks.append(excel_writer.Workbook(file))
    wb = workbooks[0]

    idx = 0
    for file in data_files:  # names of tested files
        it = iterators[0]  # nr of iterations
        opt = optimal_values[idx]  # optimal path value

        # READING A FILE WITH AN ADJACENCY MATRIX #
        global matrix, dimension
        er = 0
        data = readData.read_tsplib95(file)
        matrix = data[0]
        dimension = data[1]

        # WRITE TO OUTPUT FILE #
        worksheet = excel_write(wb, it, opt, file)

        # GENETIC ALGORITHM #
        print('\nInstance name: ' + file)
        print('Number of iterations: ' + str(it))

        for i in range(it):
            print("\nIteration: " + str(i + 1))
            start = perf_counter()  # start timing
            ga = genetic_algorithm(opt, start, er)
            best_found = ga[0]
            error = ga[1]
            end = perf_counter()  # end timinig
            elapsed = end - start
            worksheet.write(i + 4, 0, elapsed)
            worksheet.write(i + 4, 2, best_found)
            worksheet.write(i + 4, 4, error)

            print("Cost: ", best_found)
            print("Error: " + str(round(error, 2)) + "%\n")

        idx += 1
    input("Press ENTER to finish")
    wb.close()


if __name__ == "__main__":
    main()
