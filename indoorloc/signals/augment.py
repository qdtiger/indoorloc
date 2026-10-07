"""Training-time augmentations for RSSI fingerprints.

An augmentation perturbs *training* data only. The rule, which makes it safe inside
``LocalizerPipeline`` and sklearn pipelines and still natural in a torchvision-style
training loop:

* ``aug(X)`` / ``aug.augment(X)`` draw a new perturbation on every call;
* ``aug.fit_transform(X)`` fits and returns the augmented training data;
* ``aug.transform(X)`` returns ``X`` unchanged: at inference (a pipeline's ``localize``
  or ``predict``) unseen data is never perturbed.

Randomness comes only from ``random_state`` through ``np.random.default_rng``. The
generator is created on first use and advances with every call, so a fresh object with
an integer seed yields the same sequence of perturbations (``clone`` and ``load_model``
restart it). Draws depend on call order, so augmenting rows one by one differs from
augmenting the batch at once.
"""
from __future__ import annotations

import numpy as np

from . import functional as F
from .transforms import Transform, _apply


class Augmentation(Transform):
    """Base: subclasses implement ``_augment(x, rng)``; see the module docstring for the rule."""

    _real_only = True

    def _generator(self) -> np.random.Generator:
        rng = self.__dict__.get("_rng")
        if rng is None:
            rng = self._rng = np.random.default_rng(self.random_state)
        return rng

    def _get_state(self) -> dict:
        state = super()._get_state()
        state.pop("_rng", None)  # a Generator is not an array; a loaded model restarts the stream
        return state

    def augment(self, X):
        """A randomly perturbed copy of ``X`` (same container type)."""
        self._check_features(X)
        rng = self._generator()
        return _apply(lambda x: self._augment(F._real(x, type(self).__name__), rng), X)

    def transform(self, X):
        """Inference: ``X`` unchanged (augmentations never touch unseen data)."""
        self._check_features(X)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).augment(X)

    def __call__(self, X):
        return self.augment(X)

    def _augment(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class GaussianNoise(Augmentation):
    """Add zero-mean Gaussian noise of ``std_db`` dB to every heard reading (NaN stays NaN).

    Log-normal shadowing makes the dB-domain variation of RSSI around its local mean
    approximately Gaussian, so this is the textbook perturbation of a fingerprint. Apply
    it before ``FillMissing`` so that the "not heard" floor is not perturbed.

    References
        T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice Hall,
        2002, ch. 4 (log-normal shadowing).
        S. Y. Seidel, T. S. Rappaport, "914 MHz path loss prediction models for indoor wireless
        communications in multifloored buildings", IEEE Transactions on Antennas and Propagation
        40(2):207-217, 1992. https://doi.org/10.1109/8.127405
    """

    def __init__(self, std_db: float = 2.0, random_state=None):
        self.std_db = std_db
        self.random_state = random_state

    def _augment(self, x, rng):
        out = F._as_float(x)
        noise = rng.normal(0.0, float(self.std_db), size=out.shape)
        return (out + noise).astype(out.dtype)  # NaN + noise stays NaN


class APDropout(Augmentation):
    """Drop each heard reading independently with probability ``p`` (it becomes NaN).

    Simulates APs that are switched off, out of range or missed by a scan, which is the
    main source of change in long-term WiFi fingerprinting data; it is input dropout
    applied to the physical measurements rather than to network activations.

    References
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, "Dropout: a simple way
        to prevent neural networks from overfitting", Journal of Machine Learning Research 15(56):
        1929-1958, 2014. https://jmlr.org/papers/v15/srivastava14a.html
        G. M. Mendoza-Silva, P. Richter, J. Torres-Sospedra, E. S. Lohan, J. Huerta, "Long-term WiFi
        fingerprinting dataset for research on robust indoor positioning", Data 3(1):3, 2018.
        https://doi.org/10.3390/data3010003
    """

    def __init__(self, p: float = 0.1, random_state=None):
        self.p = p
        self.random_state = random_state

    def _augment(self, x, rng):
        if not 0.0 <= float(self.p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        out = F._as_float(x)
        out[rng.random(out.shape) < float(self.p)] = np.nan
        return out
