#!/usr/bin/env python3.6
import argparse
import datetime
import logging.config
import sys

from clinner.command import command
from clinner.run import Main as ClinnerMain

from pizza import Solver

logger = logging.getLogger('cli')


class Range(argparse.Action):
    def __init__(self, min=None, max=None, *args, **kwargs):
        self.min = min
        self.max = max
        kwargs["metavar"] = "[%d-%d]" % (self.min, self.max)
        super(Range, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        if not (self.min <= value <= self.max):
            msg = 'invalid choice: %r (choose from [%d-%d])' % \
                  (value, self.min, self.max)
            raise argparse.ArgumentError(self, msg)
        setattr(namespace, self.dest, value)


@command(args=((('input',), {'help': 'Input file'}),
               (('output',), {'help': 'Output file'}),
               (('-p', '--population'), {'help': 'Population size', 'type': int}),
               (('-g', '--generations'), {'help': 'Number of generations', 'type': int}),
               (('--crossover',), {'help': 'Probability to crossover', 'type': float, 'action': Range, 'min': 0, 'max': 1}),
               (('--mutation',), {'help': 'Probability to mutate', 'type': float, 'action': Range, 'min': 0, 'max': 1}),
               (('--mu',), {'help': 'Mu value', 'type': int}),
               (('--lambda',), {'help': 'Lambda value', 'type': int}),
               ),
         parser_opts={'help': 'Run the solver'})
def run(*args, **kwargs):
    before = datetime.datetime.now()
    try:
        solver = Solver(
            input_file=kwargs['input'],
            generations=kwargs['generations'],
            population_size=kwargs['population'],
            cx_probability=kwargs['crossover'],
            mutation_probability=kwargs['mutation'],
            mu=kwargs['mu'],
            lambda_=kwargs['lambda'],
        )
        solver.solve()
    except KeyboardInterrupt:
        logger.info('Interrupted')
    finally:
        elapsed_time = (datetime.datetime.now() - before).total_seconds()
        logger.debug('Time: %ss.', elapsed_time)
        # logger.info(repr(solution.best))
        # solution.write(kwargs['output'])

    return 0


class Main(ClinnerMain):
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'brief': {
                'format': '[%(levelname)s] %(message)s'
            },
            'default': {
                'datefmt': '%Y-%m-%d %H:%M:%S',
                'format': '"[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'brief',
                'level': 'DEBUG',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'filename': 'logs/app.log',
                'formatter': 'default'
            }
        },
        'loggers': {
            'hashcode_pizza': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            },
            'cli': {
                'handlers': ['console'],
                'level': 'DEBUG',
                'propagate': True
            },
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.config.dictConfig(self.LOGGING)


if __name__ == '__main__':
    sys.exit(Main().run())
