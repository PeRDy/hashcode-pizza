#!/usr/bin/env python3.6
import logging
import os
import shlex
import sys

try:
    import docker
    from docker.errors import ContainerError
    from clinner.command import command, Type as CommandType
    from clinner.run import Main
except ImportError:
    import importlib
    import pip
    import site

    print('Installing dependencies')
    pip.main(['install', '--user', '-qq', 'clinner', 'docker'])

    importlib.reload(site)

    import docker
    import jinja2
    from docker.errors import ContainerError
    from clinner.command import command, Type as CommandType
    from clinner.run import Main

logger = logging.getLogger('cli')

docker_cli = docker.from_env()

# Constants
PATH = os.path.realpath(os.path.dirname(__file__))
APP_NAME = 'hashcode-pizza'
VOLUMES = [
    f'{PATH}:/srv/apps/{APP_NAME}/app',
    f'{os.path.join(PATH, "logs")}:/srv/apps/{APP_NAME}/logs',
]


@command(command_type=CommandType.SHELL,
         args=((('--name',), {'help': 'Docker image name', 'default': APP_NAME}),
               (('--tag',), {'help': 'Docker image tag', 'default': 'latest'})),
         parser_opts={'help': 'Docker build for local environment'})
def build(*args, **kwargs):
    tag = f'{kwargs["name"]}:{kwargs["tag"]}'

    cmd = shlex.split(f'docker build -t {tag} .') + list(args)

    return [cmd]


@command(command_type=CommandType.PYTHON,
         args=((('-n', '--name',), {'help': 'Docker image name', 'default': APP_NAME}),),
         parser_opts={'help': 'Run application'})
def run(*args, **kwargs):
    try:
        container = docker_cli.containers.create(
            image=f'{APP_NAME}:latest',
            command=shlex.split('run') + list(args),
            name=kwargs['name'],
            volumes=VOLUMES,
            auto_remove=True,
        )

        container.start()

        for line in (i.decode().rstrip() for i in container.logs(stream=True)):
            print(line)
    except KeyboardInterrupt:
        docker_cli.containers.get(kwargs['name']).stop()
    except ContainerError as e:
        logger.error(e.stderr.decode())


if __name__ == '__main__':
    sys.exit(Main().run())
