"""
Downloads, processes, and stores experimental data.
Prints all data when run as a script.
"""

from collections import defaultdict
import copy
import logging
import pickle
import re
from statistics import mean
from urllib.request import urlopen

import numpy as np
import yaml

from . import cachedir, systems


class HEPData:
    """
    Interface to a `HEPData <https://hepdata.net>`_ YAML data table.

    Downloads and caches the dataset specified by the INSPIRE record and table
    number.  The web UI for `inspire_rec` may be found at
    :file:`https://hepdata.net/record/ins{inspire_rec}`.

    If `reverse` is true, reverse the order of the data table (useful for
    tables that are given as a function of Npart).

    .. note::

        Datasets are assumed to be a function of centrality.  Other kinds of
        datasets will require code modifications.

    """
    def __init__(self, inspire_rec, table, reverse=False):
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        name = 'record {} table {}'.format(inspire_rec, table)

        if cachefile.exists():
            logging.debug('loading from hepdata cache: %s', name)
            with cachefile.open('rb') as f:
                self._data = pickle.load(f)
        else:
            logging.debug('downloading from hepdata.net: %s', name)
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/yaml'.format(inspire_rec, table)
            ) as u:
                self._data = yaml.load(u)
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if reverse:
            for v in self._data.values():
                for d in v:
                    d['values'].reverse()

    @property
    def names(self):
        """
        Get the independent variable names.

        """
        data = self._data['dependent_variables']

        return [d['header']['name'] for d in data]

    def x(self, name, case=True):
        """
        Get an independent variable ("x" data) with the given name.

        If `case` is false, perform case-insensitive matching for the name.

        """
        trans = (lambda x: x) if case else (lambda x: x.casefold())
        name = trans(name)

        for x in self._data['independent_variables']:
            if trans(x['header']['name']) == name:
                return x['values']

        raise LookupError("no x data with name '{}'".format(name))

    @property
    def cent(self):
        """
        The centrality bins as a list of (low, high) tuples.

        """
        try:
            return self._cent
        except AttributeError:
            pass

        x = self.x('centrality', case=False)

        if x is None:
            raise LookupError('no centrality data')

        try:
            cent = [(v['low'], v['high']) for v in x]
        except KeyError:
            # try to guess bins from midpoints
            mids = [v['value'] for v in x]
            width = set(a - b for a, b in zip(mids[1:], mids[:-1]))
            if len(width) > 1:
                raise RuntimeError('variable bin widths')
            d = width.pop() / 2
            cent = [(m - d, m + d) for m in mids]

        self._cent = cent

        return cent

    @cent.setter
    def cent(self, value):
        """
        Manually set centrality bins.

        """
        self._cent = value

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self._data['dependent_variables']:
            if name is None or y['header']['name'].startswith(name):
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return y['values']

        raise LookupError(
            "no y data with name '{}' and qualifiers '{}'"
            .format(name, quals)
        )

    def dataset(self, name=None, maxcent=70, ignore_bins=[], **quals):
        """
        Return a dict containing:

        - **cent:** list of centrality bins
        - **x:** numpy array of centrality bin midpoints
        - **y:** numpy array of y values
        - **yerr:** subdict of numpy arrays of y errors

        `name` and `quals` are passed to `HEPData.y()`.

        Missing y values are skipped.

        Centrality bins whose upper edge is greater than `maxcent` are skipped.

        Centrality bins in `ignore_bins` [a list of (low, high) tuples] are
        skipped.

        """
        cent = []
        y = []
        yerr = defaultdict(list)

        for c, v in zip(self.cent, self.y(name, **quals)):
            # skip missing values
            # skip bins whose upper edge is greater than maxcent
            # skip explicitly ignored bins
            if v['value'] == '-' or c[1] > maxcent or c in ignore_bins:
                continue

            cent.append(c)
            y.append(v['value'])

            for err in v['errors']:
                try:
                    e = err['symerror']
                except KeyError:
                    e = err['asymerror']
                    if abs(e['plus']) != abs(e['minus']):
                        raise RuntimeError(
                            'asymmetric errors are not implemented'
                        )
                    e = abs(e['plus'])

                yerr[err.get('label', 'sum')].append(e)

        return dict(
            cent=cent,
            x=np.array([(a + b)/2 for a, b in cent]),
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
        )


def pPb5020_yield():
    """
    p going side: eta < 0
    Pb going side: eta > 0
    eta_beam = -.465

    eta_cms = eta_lab - eta_beam
    i.e. eta_cms = eta_lab + .465

    Thus if we want
    -0.5 < eta_cms < 0.5, then
    -0.5 < eta_lab - eta_beam < 0.5, and
    -0.5 + eta_beam < eta_lab < 0.5 + eta_beam

    reference: https://inspirehep.net/record/1335350

    """
    eta_beam = -0.465
    eta_cut = 1.4

    # use the V0M centrality estimator
    dset = HEPData(1335350, 2)

    # drop the last two centrality bins
    drop_bins = 2

    cent = [tuple(map(float, re.findall(r'\d+', name)))
            for name in dset.names][:-drop_bins]

    eta_lab_min, eta_lab_max = [eta + eta_beam for eta in (-eta_cut, eta_cut)]

    y, stat, sys = np.array([[
        (y['value'], y['errors'][0]['symerror'], y['errors'][1]['symerror'])
        for (x, y) in zip(dset.x('$\eta_{lab}$'), dset.y(name))
        if eta_lab_min < x['low'] and x['high'] < eta_lab_max
    ] for name in dset.names[:-drop_bins]]).T

    return dict(
        cent=cent,
        x=np.array([(a + b)/2 for a, b in cent]),
        y=y.mean(axis=0),
        yerr=dict(
            stat=np.sqrt(np.square(stat).sum(axis=0))/len(stat),
            sys=sys.mean(axis=0),
        )
    )


def pPb5020_mean_pT():
    """
    Charged particle mean pT as a function of charged particle multiplicity
    divided by mean multiplicity with mean pT in 0.15-10 GeV/c and |eta| < 0.3.

    reference: https://inspirehep.net/record/1241423

    """
    dset = HEPData(1241423, 4)
    mean_nch = 11.9

    mult = [
        tuple(round(nch/mean_nch, 3) for nch in (x['low'], x['high']))
        for x in dset.x('MULT(P=3)')
    ]

    x = np.array([
        round(0.5*(x['low'] + x['high'])/mean_nch, 3)
        for x in dset.x('MULT(P=3)')
    ])

    y, stat, sys = np.array([
        (y['value'], y['errors'][0]['symerror'], y['errors'][1]['symerror'])
        for y in dset.y('MEAN(NAME=PT)')
    ]).T

    # skip every 7th data point,
    # and drop first two bins
    nskip = 7
    ndrop = 2

    return dict(
        mult=mult[::nskip][ndrop:],
        x=x[::nskip][ndrop:],
        y=y[::nskip][ndrop:],
        yerr=dict(
            stat=stat[::nskip][ndrop:],
            sys=sys[::nskip][ndrop:],
        )
    )


def pPb5020_flows(mode):
    """
    The CMS p+Pb flows are not posted on HEP data so the data files are included
    in the git repo parent directory under expt.

    reference: https://inspirehep.net/record/1231945

    """
    # Mean Ntrk offline 0-100% centrality
    Ntrk_avg = 40.

    # Drop top-5 most central bins
    xlo, xhi, x, y, stat, sys = np.loadtxt(
        'expt/CMS_pPb5020_v{}2_sub.txt'.format(mode)
    )[:-5].T

    return dict(
        mult=list(zip(xlo/Ntrk_avg, xhi/Ntrk_avg)),
        x=x/Ntrk_avg,
        y=y,
        yerr=dict(stat=stat, sys=sys),
    )


def _data():
    """
    Curate the experimental data using the `HEPData` class and return a nested
    dict with levels

    - system
    - observable
    - subobservable
    - dataset (created by :meth:`HEPData.dataset`)

    For example, ``data['PbPb2760']['dN_dy']['pion']`` retrieves the dataset
    for pion dN/dy in Pb+Pb collisions at 2.76 TeV.

    Some observables, such as charged-particle multiplicity, don't have a
    natural subobservable, in which case the subobservable is set to `None`.

    The best way to understand the nested dict structure is to explore the
    object in an interactive Python session.

    """
    data = {s: {} for s in systems}

    if 'pPb5020' in systems:

        # pPb5020 dNch/deta
        data['pPb5020']['dNch_deta'] = {None: pPb5020_yield()}

        # pPb5020 mean pT
        data['pPb5020']['mean_pT'] = {None: pPb5020_mean_pT()}

        # pPb5020 flows
        data['pPb5020']['vnk'] = {}
        for mode in 2, 3:
            data['pPb5020']['vnk'][mode, 2] = pPb5020_flows(mode)

    if 'PbPb5020' in systems:

        # PbPb5020 dNch/deta
        name = r'$\mathrm{d}N_\mathrm{ch}/\mathrm{d}\eta$'
        data['PbPb5020']['dNch_deta'] = {None: HEPData(1410589, 2).dataset(name)}

        # PbPb5020 flows
        system, tables_nk = ('PbPb5020', [(1, [(2, 2)]), (2, [(3, 2), (4, 2)]),])

        data[system]['vnk'] = {}
        for table, nk in tables_nk:
            d = HEPData(1419244, table)
            for n, k in nk:
                data[system]['vnk'][n, k] = d.dataset(
                    'V{}{{{}{}}}'.format(
                        n, k, ', |DELTAETA|>1' if k == 2 else ''
                    ),
                    maxcent=(70 if n == 2 else 50)
                )

    return data


#: A nested dict containing all the experimental data, created by the
#: :func:`_data` function.
data = _data()


def cov(
        system, obs1, subobs1, obs2, subobs2,
        stat_frac=1e-4, sys_corr_length=100, cross_factor=.8,
        corr_obs={
            frozenset({'dNch_deta', 'dET_deta', 'dN_dy'}),
        }
):
    """
    Estimate a covariance matrix for the given system and pair of observables,
    e.g.:

    >>> cov('PbPb2760', 'dN_dy', 'pion', 'dN_dy', 'pion')
    >>> cov('PbPb5020', 'dN_dy', 'pion', 'dNch_deta', None)

    For each dataset, stat and sys errors are used if available.  If only
    "summed" error is available, it is treated as sys error, and `stat_frac`
    sets the fractional stat error.

    Systematic errors are assumed to have a Gaussian correlation as a function
    of centrality percentage, with correlation length set by `sys_corr_length`.

    If obs{1,2} are the same but subobs{1,2} are different, the sys error
    correlation is reduced by `cross_factor`.

    If obs{1,2} are different and uncorrelated, the covariance is zero.  If
    they are correlated, the sys error correlation is reduced by
    `cross_factor`.  Two different obs are considered correlated if they are
    both a member of one of the groups in `corr_obs` (the groups must be
    set-like objects).  By default {Nch, ET, dN/dy} are considered correlated
    since they are all related to particle / energy production.

    """
    def unpack(obs, subobs):
        dset = data[system][obs][subobs]
        yerr = dset['yerr']

        try:
            stat = yerr['stat']
            sys = yerr['sys']
        except KeyError:
            stat = dset['y'] * stat_frac
            sys = yerr['sum']

        return dset['x'], stat, sys

    x1, stat1, sys1 = unpack(obs1, subobs1)
    x2, stat2, sys2 = unpack(obs2, subobs2)

    if obs1 == obs2:
        same_obs = (subobs1 == subobs2)
    else:
        # check if obs are both in a correlated group
        if any({obs1, obs2} <= c for c in corr_obs):
            same_obs = False
        else:
            return np.zeros((x1.size, x2.size))

    # Use special sys error covariance matrix for p-Pb system
    if system == 'pPb5020':
        sys_corr_length = (5 if 'dNch_deta' not in [obs1, obs2] else 30)

    C = (
        np.exp(-.5*(np.subtract.outer(x1, x2)/sys_corr_length)**2) *
        np.outer(sys1, sys2)
    )

    if same_obs:
        # add stat error to diagonal
        C.flat[::C.shape[0]+1] += stat1**2
    else:
        # reduce correlation for different observables
        C *= cross_factor

    return C


def print_data(d, indent=0):
    """
    Pretty print the nested data dict.

    """
    prefix = indent * '  '
    for k in sorted(d):
        v = d[k]
        k = prefix + str(k)
        if isinstance(v, dict):
            print(k)
            print_data(v, indent + 1)
        else:
            if k.endswith('cent'):
                v = ' '.join(
                    str(tuple(int(j) if j.is_integer() else j for j in i))
                    for i in v
                )
            elif isinstance(v, np.ndarray):
                v = str(v).replace('\n', '')
            print(k, '=', v)


if __name__ == '__main__':
    print_data(data)
