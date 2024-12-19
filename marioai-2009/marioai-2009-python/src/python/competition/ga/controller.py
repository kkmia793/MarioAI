import numpy
from ga.individual import Individual

class Controller(object):
    @classmethod
    def select(cls, individuals, probabilities):
        normalized_probabilities = numpy.array(probabilities) / sum(probabilities)
        return numpy.random.choice(individuals, size=2, replace=False, p=normalized_probabilities)

    @classmethod
    def two_points_cross(cls, individual1, individual2, indices=None):
        if indices is None:
            max_index = min(len(individual1.data), len(individual2.data))
            index1 = numpy.random.randint(0, max_index // 2)
            index2 = numpy.random.randint(index1, max_index)
            indices = (index1, index2)

        child1 = numpy.hstack((individual1.data[:indices[0]],
                               individual2.data[indices[0]:indices[1]],
                               individual1.data[indices[1]:]))
        child2 = numpy.hstack((individual2.data[:indices[0]],
                               individual1.data[indices[0]:indices[1]],
                               individual2.data[indices[1]:]))

        return (Individual(data=child1), Individual(data=child2))

    @classmethod
    def mutate(cls, individuals, mutation_rate=0.05):  # 突然変異率を低減
        for individual in individuals:
            for i in range(int(len(individual.data) * mutation_rate)):
                index = numpy.random.randint(0, len(individual.data))
                individual.data[index] = numpy.random.randint(Individual.gene_size)
