import logging
import multiprocessing
from abc import ABCMeta, abstractmethod
from itertools import takewhile, zip_longest
from multiprocessing.pool import Pool
from random import randint, random
from typing import List

logger = logging.getLogger(__name__)

_solution = None


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks.

    :param iterable: Iterable to be grouped.
    :param n: Chunks length. If None, the iterable itself will be returned.
    :return: Iterable of chunks.
    """
    if n is None:
        result = (iter(iterable),)
    else:
        args = [iter(iterable)] * n
        result = (tuple(takewhile(lambda x: x is not None, i)) for i in zip_longest(*args))

    return result


class BaseSolutionSetMixin:
    @classmethod
    @abstractmethod
    def read(cls, file_path, *args, **kwargs):
        """
        Read problem instance from file.

        :param file_path: File path.
        :return: SolutionSet instance.
        """
        pass

    @abstractmethod
    def write(self, file_path, *args, **kwargs):
        """
        Write solution instance to file.

        :param file_path: File path.
        """
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run the algorithm to generate a solution.
        """
        pass


class MutateProcessOld(multiprocessing.Process):
    def __init__(self, tasks, results):
        super().__init__()
        self.tasks = tasks
        self.results = results

    def run(self):
        done = False
        while not done:
            task = self.tasks.get()
            if task is None:
                self.tasks.task_done()
                done = True
            else:
                individual = task
                individual.mutate()
                self.results.put(individual)


def mutate_process(individuals):
    return [individual.mutate() for individual in individuals]


class BreedProcess(multiprocessing.Process):
    def __init__(self, tasks, results):
        super(BreedProcess, self).__init__()
        self.tasks = tasks
        self.results = results

    def run(self):
        done = False
        while not done:
            task = self.tasks.get()
            if task is None:
                self.tasks.task_done()
                done = True
            else:
                father, mother = task
                individual = father.breed(mother)
                self.results.put(individual)


class Individual(metaclass=ABCMeta):
    @property
    @abstractmethod
    def fitness(self) -> float:
        """
        Calculate the fitness. How good is this individual.
        :return: Fitness.
        """
        return None

    @abstractmethod
    def mutate(self):
        """
        Mutate individual modifying partially the current state.
        """
        return None

    @abstractmethod
    def breed(self, mother: 'Individual') -> 'Individual':
        """
        Crossover of two individual to get a new one.
        :param mother: The second individual.
        :return: New individual
        """
        return None


class Population:
    individuals = []

    def mutate(self, rate, parents):
        """
        Mutate some individuals from this population.

        :param rate: Mutation rate.
        :param parents: Parents to mutate.
        :return: Parents mutated.
        """
        _solution = self

        with Pool() as pool:
            mutations_groups = pool.map(mutate_process, grouper(parents, (len(parents)//4)+1))
            mutations = [m for g in mutations_groups for m in g]

        return mutations

    def breed(self, parents):
        """
        Breed the rest of the population.

        :param parents: Current parents.
        :return: Children, new individuals.
        """
        parents_length = len(parents)
        children = []
        while parents_length + len(children) < len(self.individuals):
            father = randint(0, parents_length - 1)
            mother = randint(0, parents_length - 1)
            while mother == father:
                mother = randint(0, parents_length - 1)

            children.append(parents[father].breed(parents[mother]))

        return children

    def evolve(self, retain: float, random_select: float, mutate: float) -> List[Individual]:
        """
        Evolve a population applying following steps:
        1. Retain the best performing individuals.
        2. Randomly select some other individuals.
        3. Mutate individuals from 1 and 2.
        4. Fill the population with breeding between elements from 1 to 3.

        :param retain: Retaining rate.
        :param random_select: Randomly selection rate.
        :param mutate: Mutation rate.
        :return: Individuals from evolved population.
        """
        # Sort individuals by fitness
        individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

        # Retain a percentage of best performing individuals.
        parents = individuals[:int(len(individuals) * retain)]
        individuals = individuals[len(parents):]

        # Randomly select other individuals to promote genetic diversity
        parents += [i for i in individuals if random_select > random()]

        # Mutate some individuals
        parents = self.mutate(mutate, parents)

        # Crossover parents to create the rest of children
        children = self.breed(parents)

        parents += children

        return sorted(parents, key=lambda x: x.fitness, reverse=True)

    def run(self, threshold: float = 0.9, epochs: int = 100, retain: float = 0.2, select: float = 0.05,
            mutate: float = 0.01):
        """
        Run the evolution to find a solution. The evolution will stop when a individual achieves a fitness value above
        a threshold or when a number of epochs concludes.

        :param threshold: Threshold to stop evolution.
        :param epochs: Maximum number of epochs.
        :param retain: Evolution retaining rate.
        :param select: Evolution randomly selection rate.
        :param mutate: Evolution mutation rate.
        """
        parameters = {
            'Maximum number of epochs': f'{epochs}',
            'Threshold to stop evolution': f'{threshold * 100:.2f}%',
            'Retain': f'{retain * 100:.2f}%',
            'Select': f'{select * 100:.2f}%',
            'Mutate': f'{mutate * 100:.2f}%',
        }
        msg = 'Genetic Algorithm parameters:\n'
        msg += '\n'.join([f'{name:>27}: {value:>6}' for name, value in parameters.items()])
        logger.info(msg)
        e = 0
        epochs_to_print = range(0, epochs, epochs // 10)
        while threshold > self.best.fitness and e < epochs:
            self.individuals = self.evolve(retain, select, mutate)
            if e in epochs_to_print:
                logger.info('Epoch %d, %r', e, self.best)
            e += 1

    @property
    def best(self):
        """
        Best individual.
        """
        return self.individuals[0]
