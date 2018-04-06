"""
Trains Gaussian process emulators.

When run as a script, allows retraining emulators, specifying the number of
principal components, and other options (however it is not necessary to do this
explicitly --- the emulators will be trained automatically when needed).  Run
``python -m src.emulator --help`` for usage information.

Uses the `scikit-learn <http://scikit-learn.org>`_ implementations of
`principal component analysis (PCA)
<http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
and `Gaussian process regression
<http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""

import logging
import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import QuantileTransformer

from . import cachedir, lazydict, model
from .design import Design


class _Covariance:
    """
    Proxy object to extract observable sub-blocks from a covariance array.
    Returned by Emulator.predict().

    """
    def __init__(self, array, slices):
        self.array = array
        self._slices = slices

    def __getitem__(self, key):
        (obs1, subobs1), (obs2, subobs2) = key
        return self.array[
            ...,
            self._slices[obs1][subobs1],
            self._slices[obs2][subobs2]
        ]


class Emulator:
    """
    Multidimensional Gaussian process emulator using principal component
    analysis.

    The model training data are standardized (subtract mean and scale to unit
    variance), then transformed through PCA.  The first `npc` principal
    components (PCs) are emulated by independent Gaussian processes (GPs).  The
    remaining components are neglected, which is equivalent to assuming they
    are standard zero-mean unit-variance GPs.

    This class has become a bit messy but it still does the job.  It would
    probably be better to refactor some of the data transformations /
    preprocessing into modular classes, to be used with an sklearn pipeline.
    The classes would also need to handle transforming uncertainties, which
    could be tricky.

    """
    # observables to emulate
    # list of 2-tuples: (obs, [list of subobs])

    PbPb5020 = [
        ('dNch_deta', [None]),
        ('vnk', [(2, 2), (3, 2), (4, 2)]),
    ]

    pPb5020 = [
        ('dNch_deta', [None]),
        ('mean_pT', [None]),
        ('vnk', [(2, 2), (3, 2)]),
    ]

    def __init__(self, system, npc=10, nrestarts=0):
        logging.info(
            'training emulator for system %s (%d PC, %d restarts)',
            system, npc, nrestarts
        )

        Y = []
        self._slices = {}

        # system specific observables
        observables = {
            'pPb5020': self.pPb5020,
            'PbPb5020': self.PbPb5020,
        }[system]

        # Build an array of all observables to emulate.
        nobs = 0
        for obs, subobslist in observables:
            self._slices[obs] = {}
            for subobs in subobslist:
                Y.append(model.data[system][obs][subobs]['Y'])
                n = Y[-1].shape[1]
                self._slices[obs][subobs] = slice(nobs, nobs + n)
                nobs += n

        Y = np.concatenate(Y, axis=1)

        self.npc = npc
        self.nobs = nobs

        # Quantile transform observables to unit normal distribution
        self.pt = QuantileTransformer(
            copy=False, n_quantiles=500, output_distribution='normal'
        )

        # Principal-component transformation
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')

        # Transform observables through PCA.  Use the first `npc`
        # components but save the full PC transformation for later.
        Z = self.pca.fit_transform(self.pt.fit_transform(Y))[:, :npc]

        # Define kernel (covariance function):
        # Gaussian correlation (RBF) plus a noise term.
        design = Design(system)
        ptp = design.max - design.min
        kernel = (
            1. * kernels.RBF(
                length_scale=ptp,
                length_scale_bounds=np.outer(ptp, (.1, 10))
            ) +
            kernels.WhiteKernel(
                noise_level=.1**2,
                noise_level_bounds=(.01**2, 1)
            )
        )

        # Fit a GP (optimize the kernel hyperparameters) to each PC.
        self.gps = [
            GPR(
                kernel=kernel, alpha=0,
                n_restarts_optimizer=nrestarts,
                copy_X_train=False
            ).fit(design, z)
            for z in Z.T
        ]

        # Construct the full linear transformation matrix, which is just the PC
        # matrix with the first axis multiplied by the explained standard
        # deviation of each PC and the second axis multiplied by the
        # standardization scale factor of each observable.
        self._trans_matrix = (
            self.pca.components_
            * np.sqrt(self.pca.explained_variance_[:, np.newaxis])
        )

    @classmethod
    def from_cache(cls, system, retrain=False, **kwargs):
        """
        Load the emulator for `system` from the cache if available, otherwise
        train and cache a new instance.

        """
        cachefile = cachedir / 'emulator' / '{}.pkl'.format(system)

        # cache the __dict__ rather than the Emulator instance itself
        # this way the __name__ doesn't matter, e.g. a pickled
        # __main__.Emulator can be unpickled as a src.emulator.Emulator
        if not retrain and cachefile.exists():
            logging.debug('loading emulator for system %s from cache', system)
            emu = cls.__new__(cls)
            emu.__dict__ = joblib.load(cachefile)
            return emu

        emu = cls(system, **kwargs)

        logging.info('writing cache file %s', cachefile)
        cachefile.parent.mkdir(exist_ok=True)
        joblib.dump(emu.__dict__, cachefile, protocol=pickle.HIGHEST_PROTOCOL)

        return emu

    def _inverse_transform(self, Z, array=False):
        """
        Inverse transform principal components to observables.

        Returns a nested dict of arrays.

        """
        # Z shape (..., npc)
        # Y shape (..., nobs)
        Y = np.dot(Z, self._trans_matrix[:Z.shape[-1]])

        shape = Y.shape
        Y = self.pt.inverse_transform(Y.reshape(-1, self.nobs)).reshape(shape)

        if array:
            return Y

        return {
            obs: {
                subobs: Y[..., s]
                for subobs, s in slices.items()
            } for obs, slices in self._slices.items()
        }

    def predict(self, X, return_cov=False, extra_std=0):
        """
        Predict y at location x using the emulator

        """
        gp_mean = [gp.predict(X, return_cov=False) for gp in self.gps]

        mean = self._inverse_transform(
            np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1)
        )

        if return_cov:
            Y = self.sample_y(X, n_samples=10**3, array=True)
            cov = np.array([np.cov(y, rowvar=False) for y in Y])
            return mean, _Covariance(cov, self._slices)
        else:
            return mean

    def sample_y(self, X, n_samples=1, random_state=None, array=False):
        """
        Sample model output at `X`.

        Returns a nested dict of observable arrays, each with shape
        ``(n_samples_X, n_samples, n_cent_bins)``.

        """
        # Sample the GP for each emulated PC.  The remaining components are
        # assumed to have a standard normal distribution.
        return self._inverse_transform(
            np.concatenate([
                gp.sample_y(
                    X, n_samples=n_samples, random_state=random_state
                )[:, :, np.newaxis]
                for gp in self.gps
            ] + [
                np.random.standard_normal(
                    (X.shape[0], n_samples, self.pca.n_components_ - self.npc)
                )
            ], axis=2),
            array=array
        )


emulators = lazydict(Emulator.from_cache)


if __name__ == '__main__':
    import argparse
    from . import systems

    def arg_to_system(arg):
        if arg not in systems:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(
        description='train emulators for each collision system',
        argument_default=argparse.SUPPRESS
    )

    parser.add_argument(
        '--npc', type=int,
        help='number of principal components'
    )
    parser.add_argument(
        '--nrestarts', type=int,
        help='number of optimizer restarts'
    )

    parser.add_argument(
        '--retrain', action='store_true',
        help='retrain even if emulator is cached'
    )
    parser.add_argument(
        'systems', nargs='*', type=arg_to_system,
        default=systems, metavar='SYSTEM',
        help='system(s) to train'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    for s in kwargs.pop('systems'):
        emu = Emulator.from_cache(s, **kwargs)

        print(s)
        print('{} PCs explain {:.5f} of variance'.format(
            emu.npc,
            emu.pca.explained_variance_ratio_[:emu.npc].sum()
        ))

        for n, (evr, gp) in enumerate(zip(
                emu.pca.explained_variance_ratio_, emu.gps
        )):
            print(
                'GP {}: {:.5f} of variance, LML = {:.5g}, kernel: {}'
                .format(n, evr, gp.log_marginal_likelihood_value_, gp.kernel_)
            )
