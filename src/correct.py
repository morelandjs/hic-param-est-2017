"""
Module which includes functions to correct specific problems that arise
in a combined analysis of p-Pb and Pb-Pb collisions at 5.02 TeV.

This code does not readily generalize to other projects.

"""
from functools import partial
import logging
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import curve_fit

from . import cachedir
from .design import Design


import matplotlib.pyplot as plt


def run_cmd(*args):
    """
    Run and log a subprocess.
    """
    cmd = ' '.join(args)
    logging.info('running command: %s', cmd)

    try:
        proc = subprocess.run(
            cmd.split(), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status %d:\n%s',
            e.returncode, e.output.strip('\n')
        )
        raise
    else:
        logging.debug(
            'command completed successfully:\n%s',
            proc.stdout.strip('\n')
        )
        return proc


def trento(args, nevents=10**5):
    """
    Runs trento and save each event's initial entropy

    """
    cachefile = Path(cachedir, 'trento', args['filename'])
    cachefile.parent.mkdir(parents=True, exist_ok=True)

    proc = run_cmd(
        'trento {}'.format(args['projectiles']),
        '--number-events {}'.format(nevents),
        '--normalization {}'.format(args['norm']),
        '--reduced-thickness {}'.format(args['trento_p']),
        '--fluctuation {}'.format(args['fluctuation']),
        '--nucleon-width {}'.format(args['nucleon_width']),
        '--parton-width {}'.format(args['parton_width']),
        '--parton-number {}'.format(args['parton_number'].astype(int)),
        '--nucleon-min-dist {}'.format(args['dmin']),
        '--cross-section {}'.format(args['cross_section']),
        '--grid-step {} --grid-max {}'.format(.5*args['parton_width'], 10),
    )

    np.save(cachefile, [float(l.split()[3]) for l in proc.stdout.splitlines()])


def trento_args(system, design_points):
    """
    Generator which yields trento arguments at each design point.

    """
    design = Design(system)
    param = dict(zip(design.points, design.array))

    for design_point in design_points:
        args = dict(zip(design.keys, param[design_point]))

        parton_number = args['parton_number'].astype(int)
        nucleon_width = args['nucleon_width']
        parton_struct = args['parton_struct']
        sigma = args['fluct_std']
        dmin3 = args['dmin3']
        min_size = .2

        args['filename'] = Path(system, design_point)
        args['projectiles'] = ' '.join(design.projectiles)
        args['parton_width'] = min_size + parton_struct*(nucleon_width - min_size)
        args['fluctuation'] = 1/(sigma**2 * parton_number)
        args['dmin'] = dmin3**(1/3)
        args['cross_section'] = {
            200: 4.2, 2760: 6.4, 5020: 7.0
        }[design.beam_energy]

        yield args


def trento_entropy(system, design_point):
    """
    Return the entropy of each trento event in a large minimum bias sample.

    """
    design = Design(system)

    cachefiles = [
        Path(cachedir, 'trento', system, '{}.npy'.format(p))
        for p in design.points
    ]

    cachefile = Path(cachedir, 'trento', system, '{}.npy'.format(design_point))

    # if all cache files exist, load trento entropy from cache
    if all(f.exists() for f in cachefiles):
        return np.load(cachefile)

    # otherwise generate missing cache files
    missing_design_points = [
        p for (f, p) in zip(cachefiles, design.points) if not f.exists()
    ]

    # run trento events
    ncpu = multiprocessing.cpu_count()
    multiprocessing.Pool(ncpu).map(
        trento, trento_args(system, missing_design_points)
    )

    return np.load(cachefile)


def correct_centrality(system, design_point, events, bins):
    """
    Divide events into centrality bins according to initial entropy
    cuts. This method is superior to simply sorting the events by
    dNch/deta when the number of events in the sample is small.

    """
    entropy = np.sort(trento_entropy(system, design_point))
    n = entropy.size

    binned_entropy = [
        entropy[int((1 - b/100)*n):int((1 - a/100)*n)]
        for a, b in bins
    ]

    entropy_bins = [(min(s), max(s)) for s in binned_entropy]
    s = events['init_entropy']

    return [events[(slo < s) & (s < shi)] for slo, shi in entropy_bins]


def powerlaw(x, a, b, c):
    """
    Charged particle multiplicity as a function of initial entropy density
    empirically follows a power law relation.

    The constant c accounts for missing entropy below the UrQMD switching
    temperature. This is commonly referred to as the "corona".

    """
    return a*x**b + c


def fit_powerlaw(events):
    """
    Fit a powerlaw for (dNch/deta)(entropy) using a large sample of events.

    """
    events.sort(order='init_entropy')

    x, y = [events[k] for k in ('init_entropy', 'dNch_deta')]
    nonzero = (x > 0) & (y > 0)

    (a, b, c), pcov = curve_fit(powerlaw, x[nonzero], y[nonzero])

    return partial(powerlaw, a=a, b=b, c=c)


def correct_yield(events):
    """
    Correct an individual event's dNch/deta to account for missing entropy
    below the Cooper-Frye particlization temperature.

    """
    try:
        interp_nch = fit_powerlaw(events)
    except RuntimeError:
        return events

    for n, event in enumerate(events):
        s, nch = [event[k] for k in ('init_entropy', 'dNch_deta')]
        nch_missing = max(-interp_nch(0), 0)
        nch = nch if nch > 0 else interp_nch(s)
        events[n]['dNch_deta'] = nch + nch_missing

    return events
