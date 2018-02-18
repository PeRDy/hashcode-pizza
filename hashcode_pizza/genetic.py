import logging
import multiprocessing
from abc import ABCMeta, abstractmethod
from random import randint, random
from typing import List

logger = logging.getLogger(__name__)


class MutateProcess(multiprocessing.Process):
    def __init__(self, tasks, results):
        super(MutateProcess, self).__init__()
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

    def mutate_multiprocessing(self, rate, parents):
        """
        Mutate some individuals from this population.

        :param rate: Mutation rate.
        :param parents: Parents to mutate.
        :return: Parents mutated.
        """
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.JoinableQueue()

        processes = [MutateProcess(tasks, results) for _ in range(multiprocessing.cpu_count())]

        for w in processes:
            w.start()

        not_mutated = []
        for individual in parents:
            if rate > random():
                tasks.put(individual)
            else:
                not_mutated.append(individual)

        for _ in range(multiprocessing.cpu_count()):
            tasks.put(None)

        tasks.join()

        num_mutated = len(parents) - len(not_mutated)
        mutated = []
        for _ in range(num_mutated):
            mutated.append(results.get())
            results.task_done()

        for p in processes:
            p.join()

        tasks.close()
        results.close()

        return not_mutated + mutated

    def mutate(self, rate, parents):
        """
        Mutate some individuals from this population.

        :param rate: Mutation rate.
        :param parents: Parents to mutate.
        :return: Parents mutated.
        """
        return [individual.mutate() if rate > random() else individual for individual in parents]

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

            if father != mother:
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
        self.mutate(mutate, parents)

        # Crossover parents to create the rest of children
        children = self.breed(parents)

        parents += children

        return sorted(parents, key=lambda x: x.fitness, reverse=True)

    def run(self, threshold: float = 0.9, epochs: int = 100, retain: float = 0.2, random_select: float = 0.05,
            mutate: float = 0.01, *args, **kwargs):
        """
        Run the evolution to find a solution. The evolution will stop when a individual achieves a fitness value above
        a threshold or when a number of epochs concludes.

        :param threshold: Threshold to stop evolution.
        :param epochs: Maximum number of epochs.
        :param retain: Evolution retaining rate.
        :param random_select: Evolution randomly selection rate.
        :param mutate: Evolution mutation rate.
        """
        parameters = {
            'Maximum number of epochs': f'{epochs}',
            'Threshold to stop evolution': f'{threshold * 100:.0f}%',
            'Retain': f'{retain * 100:.0f}%',
            'Select': f'{random_select * 100:.0f}%',
            'Mutate': f'{mutate * 100:.0f}%',
        }
        msg = 'Genetic Algorithm parameters:\n'
        msg += '\n'.join([f'{name:>27}: {value:>5}' for name, value in parameters.items()])
        logger.info(msg)
        e = 0
        epochs_to_print = range(0, epochs, epochs // 10)
        while threshold > self.best.fitness and e < epochs:
            self.individuals = self.evolve(retain, random_select, mutate)
            if e in epochs_to_print:
                logger.info('Epoch %d, %r', e, self.best)
            e += 1

    @property
    def best(self):
        """
        Best individual.
        """
        return self.individuals[0]

    @classmethod
    @abstractmethod
    def read(cls, file_path, population: int):
        pass

    @abstractmethod
    def write(self, file_path):
        pass
