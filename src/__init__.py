""" Project initialization and common objects. """

import copy
import logging
import os
from pathlib import Path
import re
import sys

import numpy as np


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = workdir / 'cache'
cachedir.mkdir(parents=True, exist_ok=True)

#: Sets the collision systems for the entire project,
#: where each system is a string of the form
#: ``'<projectile 1><projectile 2><beam energy in GeV>'``,
#: such as ``'PbPb2760'``, ``'AuAu200'``, ``'pPb5020'``.
#: Even if the project uses only a single system,
#: this should still be a list of one system string.
systems = ['pPb5020', 'PbPb5020']


def parse_system(system):
    """
    Parse a system string into a pair of projectiles and a beam energy.

    """
    match = re.fullmatch('([A-Z]?[a-z])([A-Z]?[a-z])([0-9]+)', system)
    return match.group(1, 2), int(match.group(3))


class lazydict(dict):
    """
    A dict that populates itself on demand by calling a unary function.

    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __missing__(self, key):
        self[key] = value = self.function(key, *self.args, **self.kwargs)
        return value


def transform(data):
    """
    Log transform model or experimental data,

    y = log y,
    d(log y) = dy/y.

    This transformation improves PCA decomposition and emulator performance.

    """
    if 'x' in data:
        d = {}
        for k, v in data.items():
            if k in ['y', 'Y']:
                d[k] = np.log(v)
            elif k == 'yerr':
                d[k] = {s: data['yerr'][s]/data['y'] for s in ['stat', 'sys']}
            else:
                d[k] = v
        return d
    else:
        return {k: transform(v) for k, v in data.items()}
