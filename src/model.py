""" model output """

import logging
from pathlib import Path
import pickle

from hic import flow
import numpy as np
from sklearn.externals import joblib

from . import workdir, cachedir, systems, lazydict, expt
from .design import Design

import matplotlib.pyplot as plt

def pT_fluct(events):
    """
    Compute the relative mean pT fluctuation \sqrt{C_m}/M(p_T)_m, defined in
    Eqs. (2-3) of the ALICE paper https://inspirehep.net/record/1307102.

    """
    N, sum_pT, sum_pTsq = [
        events['pT_fluct'][k] for k in
        ['N', 'sum_pT', 'sum_pTsq']
    ]

    Npairs = .5*N*(N - 1)
    M = sum_pT.sum() / N.sum()

    # This is equivalent to the sum over pairs in Eq. (2).  It may be derived
    # by using that, in general,
    #
    #   \sum_{i,j>i} a_i a_j = 1/2 [(\sum_{i} a_i)^2 - \sum_{i} a_i^2].
    #
    # That is, the sum over pairs (a_i, a_j) may be re-expressed in terms of
    # the sum of a_i and sum of squares a_i^2.  Applying this to Eq. (2) and
    # collecting terms yields the following expression.
    C = (
        .5*(sum_pT**2 - sum_pTsq) - M*(N - 1)*sum_pT + M**2*Npairs
    ).sum() / Npairs.sum()

    return np.sqrt(C)/M


# TODO move this symmetric cumulant code to hic

def csq(x):
    """
    Return the absolute square |x|^2 of a complex array.

    """
    return (x*x.conj()).real


def corr2(Qn, N):
    """
    Compute the two-particle correlation <v_n^2>.

    """
    return (csq(Qn) - N).sum() / (N*(N - 1)).sum()


def symmetric_cumulant(events, m, n):
    """
    Compute the symmetric cumulant SC(m, n).

    """
    N = np.asarray(events['flow']['N'], dtype=float)
    Q = dict(enumerate(events['flow']['Qn'].T, start=1))

    cm2n2 = (
        csq(Q[m]) * csq(Q[n])
        - 2*(Q[m+n] * Q[m].conj() * Q[n].conj()).real
        - 2*(Q[m] * Q[m-n].conj() * Q[n].conj()).real
        + csq(Q[m+n]) + csq(Q[m-n])
        - (N - 4)*(csq(Q[m]) + csq(Q[n]))
        + N*(N - 6)
    ).sum() / (N*(N - 1)*(N - 2)*(N - 3)).sum()

    cm2 = corr2(Q[m], N)
    cn2 = corr2(Q[n], N)

    return cm2n2 - cm2*cn2


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'


class ModelData:
    """
    Helper class for event-by-event model data.  Reads binary data files and
    computes centrality-binned observables.

    """
    species = ['charged', 'pion', 'kaon', 'proton', 'Lambda', 'Sigma0', 'Xi', 'Omega']

    dtype = np.dtype([
        ('trigger', (float_t, 2)),
        ('initial_entropy', float_t),
        ('nsamples', int_t),
        ('dNch_deta', float_t),
        ('dET_deta', float_t),
        ('dN_dy', [(s, float_t) for s in species]),
        ('mean_pT', [(s, float_t) for s in species]),
        ('pT_fluct', [('N', int_t), ('sum_pT', float_t), ('sum_pTsq', float_t)]),
        ('flow', [('N', int_t), ('Qn', complex_t, 8)]),
    ])

    def __init__(self, *files):
        # read each file using the above dtype
        def load_events(f):
            logging.debug('loading %s', f)
            d = np.fromfile(str(f), dtype=self.dtype)
            d.sort(order='dNch_deta')
            return d

        self.events = [load_events(f) for f in files]

    def observables_like(self, data, *keys):
        """
        Compute the same centrality-binned observables as contained in `data`
        with the same nested dict structure.

        This function calls itself recursively, each time prepending to `keys`.

        """
        try:
            x = data['x']
            bin_type = [k for k in data.keys() if k in ('mult', 'cent')].pop()
            bins = data[bin_type]
        except KeyError:
            return {
                k: self.observables_like(v, k, *keys)
                for k, v in data.items()
            }

        def _compute_bin():
            """
            Choose a function to compute the current observable for a single
            centrality bin.

            """
            obs_stack = list(keys)
            obs = obs_stack.pop()

            if obs in ['dNch_deta', 'dET_deta']:
                return lambda events: events[obs].mean()

            if obs == 'dN_dy':
                species = obs_stack.pop()
                return lambda events: events[obs][species].mean()

            if obs == 'mean_pT':
                species = obs_stack.pop()
                def functor(events):
                    if sum(events['dN_dy'][species]) > 0:
                        return np.average(
                            events[obs][species],
                            weights=(events['dN_dy'][species])
                        )
                    return 0
                return functor

            if obs == 'pT_fluct':
                return pT_fluct

            if obs.startswith('vnk'):
                nk = obs_stack.pop()
                return lambda events: flow.Cumulant(
                    events['flow']['N'], *events['flow']['Qn'].T[1:]
                ).flow(*nk, imaginary='zero')

            if obs.startswith('sc'):
                mn = obs_stack.pop()
                return lambda events: symmetric_cumulant(events, *mn)

        compute_bin = _compute_bin()

        def compute_all_bins(events):

            trigger = events['trigger']
            minbias = (0, float('inf'))

            if bin_type == 'cent':
                minbias_events = events[[all(b) for b in (trigger == minbias)]]
                n = minbias_events.size
                binned_events = [
                    minbias_events[int((1 - b/100)*n):int((1 - a/100)*n)]
                    for a, b in bins
                ]
            elif bin_type == 'mult':
                binned_events = [
                    events[(abs(events['trigger'] - mbin) < 1e-3).all(axis=1)]
                    for mbin in bins
                ]
            else:
                raise ValueError("no such bin type")

            return list(map(compute_bin, binned_events))

        Y = np.array(list(map(compute_all_bins, self.events))).squeeze()

        return {'x': x, bin_type: bins, 'Y': Y}


def _data(system, validation=False):
    """
    Compute training or validation data (model observables at all design
    points) for the given system.

    """
    design = Design(system, validation=validation)

    # expected filenames for each design point
    files = [
        Path(workdir, 'model_output', design.type, system, '{}.dat'.format(p))
        for p in design.points
    ]

    cachefile = Path(cachedir, 'model', design.type, '{}.pkl'.format(system))

    if cachefile.exists():
        # use the cache unless any of the model data files are newer
        # this DOES NOT check any other logical dependencies, e.g. the
        # experimental data
        # to force recomputation, delete the cache file
        mtime = cachefile.stat().st_mtime
        if all(f.stat().st_mtime < mtime for f in files):
            logging.debug('loading observables cache file %s', cachefile)
            return joblib.load(cachefile)
        else:
            logging.debug('cache file %s is older than event data', cachefile)
    else:
        logging.debug('cache file %s does not exist', cachefile)

    logging.info(
        'loading %s/%s data and computing observables',
        system, design.type
    )

    data = expt.data[system]

    # Add dummy expt data for PbPb5020 mean pT so that
    # the model predictions are still calculated.
    if system.startswith('PbPb'):
        bins = np.linspace(0, 80, 9)
        cent = [(a, b) for a, b in zip(bins[:-1], bins[1:])]
        mean_pT = dict(
            cent=cent,
            x=np.array([(a + b)/2 for a, b in cent]),
        )
        data = dict(mean_pT=dict(charged=mean_pT), **data)

    data = ModelData(*files).observables_like(data)

    logging.info('writing cache file %s', cachefile)
    cachefile.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, cachefile, protocol=pickle.HIGHEST_PROTOCOL)

    return data


data = lazydict(_data)
validation_data = lazydict(_data, validation=True)


if __name__ == '__main__':
    from pprint import pprint
    for s in systems:
        print(s)
        d = data[s]
        pprint(d)
