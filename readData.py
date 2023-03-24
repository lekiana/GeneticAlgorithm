import configparser
import tsplib95


def read_tsplib95(file_name):
    problem = tsplib95.load(file_name)
    dimension = problem.dimension
    rows, cols = (dimension, dimension)
    matrix = [[0 for i in range(cols)] for j in range(rows)]
    for i in range(1, dimension):
        for j in range(1, dimension):
            edge = i, j
            weight = problem.get_weight(*edge)
            matrix[i][j] = weight
    return matrix, dimension


def read_config_section(config_name, section_name):
    config = configparser.ConfigParser()
    config.read(config_name)
    output = list()
    section = config[section_name]
    for key, value in section.items():
        output.append(value)
    return output


def read_config():
    data_files = read_config_section('config.ini', 'FILE')
    it_read = read_config_section('config.ini', 'ITERATOR')
    iterators = [int(x) for x in it_read]
    opt_read = read_config_section('config.ini', 'OPTIMAL_VALUE')
    optimal_values = [int(x) for x in opt_read]
    optimal_paths = []
    output_files = read_config_section('config.ini', 'OUTPUT')
    return data_files, iterators, optimal_values, optimal_paths, output_files


def read_data(file_name):
    f = open(file_name)
    # read dimension #
    while True:
        line = f.readline()
        if line.find('DIMENSION: ') != -1:
            idx = line.find('DIMENSION: ') + 11
            break
    size = line[idx]
    while line[idx] != '\n':
        size += line[idx + 1]
        idx += 1

    # read edge weight section #
    while True:
        line = f.readline()
        if line.find('EDGE_WEIGHT_SECTION') != -1:
            break

    size = int(size)
    rows, cols = (size, size)
    matrix = [[0 for i in range(cols)] for j in range(rows)]
    idx = 0
    weight = ''
    line = f.read()
    for r in range(size):
        for c in range(size):
            while line[idx] == ' ' or line[idx] == '\n':
                idx += 1
            while line[idx] != ' ' and line[idx] != '\n':
                weight += line[idx]
                idx += 1
            matrix[r][c] = int(weight)
            weight = ''

    return matrix, size