# L3 methods

`indoorloc.methods` holds the localization methods behind one scikit-learn-style contract. Only
numpy is imported at module level. The scikit-learn, torch and skada backed methods import those
packages when they are fitted or loaded.

[Guide home](index.md) · [中文](../zh/methods.md)

## The estimator contract

| Call | Meaning |
| --- | --- |
| `create_model(name, *, preprocess=None, **params)` | a method by registry name (or `"package.module:Class"`); `preprocess=` wraps it with an L2 transform in a `LocalizerPipeline` |
| `fit(X, y=None, *, floor=None, building=None, **fit_params)` | `X` is `(N, ...)` and `y` is `(N, D)` positions, or `X` is a `SampleTable` that supplies `y`, `floor` and `building` |
| `localize(X) -> Prediction` | positions plus floor, building, ids and `spread` |
| `predict(X) -> ndarray` | positions only, as in scikit-learn |
| `evaluate(table)` or `evaluate(X, y, floor=...)` | an L4 `EvaluationResults`; samples the method could not place are left out of the errors and counted in `n_failed` |
| `score(X, y)` | minus the mean position error, so `GridSearchCV` maximises it; NaN when any sample cannot be placed (`evaluate(...).n_failed` counts them) |
| `save(path)`, `iloc.load_model(path)` | `config.json` + `arrays.npz`, no pickle |
| `get_params`, `set_params`, `iloc.clone` | scikit-learn semantics; `sklearn.base.clone` also works |

Rules every method follows:

* `__init__` only stores parameters. Learned attributes end with `_`, and a failed `fit` leaves
  the model unfitted.
* Missing readings are NaN. A method that cannot use NaN refuses it and names the fix
  (`preprocess=FillMissing(-104)`); the NaN input column of the catalog says which methods accept NaN.
* Results are deterministic. Ties are broken by training index, and randomness comes only from
  `random_state`. The k-NN search re-scores its candidates with distances summed in a fixed order
  (exact integers for whole-dBm readings) and breaks equal distances by training index, so its
  result does not depend on BLAS threads or batch size.
  This is why the UJIIndoorLoc numbers of 0.2 differ from those of 0.1 (see
  [MIGRATION.md](../../MIGRATION.md#why-the-ujiindoorloc-numbers-changed)).
* Floor and building are predicted when the training data has them (vote, classifier or head).
* Model-based methods (trilateration, TDoA, AoA, path loss, weighted centroid, VLC) take the anchor
  geometry as `anchors=` `(A, D)`; `fit` only records the input size or calibrates biases and
  path-loss parameters. `localize` works before `fit`, unless `calibrate=True` (or, for path
  loss, a parameter is left to be learned); see [Model-based](#model-based).

```python
import numpy as np
import indoorloc as iloc
from indoorloc.signals import FillMissing

train, test = iloc.load_dataset("synthetic_office")
model = iloc.create_model("wknn", k=5, preprocess=FillMissing(-104)).fit(train)
pred = model.localize(test)
print(pred.pos.shape, pred.floor[:3], model.predict(test.X[:2]).round(2).tolist())
# (200, 2) [0 0 0] [[13.0, 14.15], [1.78, 17.3]]
model.save("wknn_model")                         # a folder: config.json + arrays.npz
again = iloc.load_model("wknn_model")
print(np.array_equal(again.predict(test.X), model.predict(test.X)))
# True
```

### scikit-learn interoperability

Models are scikit-learn estimators. Nested parameters such as `localizer__k` work in grid
searches. Group folds over reference points keep scans taken at one point on the same side of
each split:

```python
# requires: sklearn
from sklearn.model_selection import GridSearchCV, GroupKFold

X, y = train.to_numpy()
search = GridSearchCV(iloc.create_model("knn", preprocess=FillMissing(-104)),
                      {"localizer__k": [1, 3, 5, 9], "localizer__weights": ["uniform", "distance"]},
                      cv=GroupKFold(5))
search.fit(X, y, groups=train.groups["point"])
print(search.best_params_, round(-search.best_score_, 3))
# {'localizer__k': 9, 'localizer__weights': 'distance'} 2.628
```

### Registering your own method

A subclass of `BaseLocalizer` implements `_fit` and `_localize`. `register_model` makes it
available to `create_model` and to the command line. [Extending](extending.md#a-method)
shows a complete example.

## Catalog

Generated from `indoorloc.methods.METHODS` and the class docstrings by `python docs/catalog.py`.
The Extra column names the pip extra a method needs beyond numpy.

<!-- catalog:methods -->
**Fingerprinting**

| Name | What it does | Extra | NaN input | Reference |
| --- | --- | --- | --- | --- |
| [`ensemble`](../../indoorloc/methods/ensemble.py)<br>`EnsembleLocalizer` | Averaging ensemble: combine the positions of several localizers, vote their labels. | numpy only | as its members | Dietterich, T. G., "Ensemble methods in machine learning", Multiple Classifier Systems (MCS) 2000, LNCS 1857. [doi:10.1007/3-540-45014-9_1](https://doi.org/10.1007/3-540-45014-9_1) |
| [`extratrees`](../../indoorloc/methods/sklearn_wrap.py)<br>`ExtraTreesLocalizer` | Extremely randomized trees: RandomForestLocalizer with random split thresholds. | `[sklearn]` | no (fill first) | Geurts, P., Ernst, D., Wehenkel, L., "Extremely randomized trees", Machine Learning 63(1), 2006. [doi:10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1) |
| [`gbdt`](../../indoorloc/methods/sklearn_wrap.py)<br>`GradientBoostingLocalizer` | Histogram gradient-boosted trees: one booster per coordinate axis, boosted classifiers for labels. | `[sklearn]` | yes | Friedman, J. H., "Greedy function approximation: A gradient boosting machine", Annals of Statistics 29(5), 2001. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) (+1 more in the docstring) |
| [`gp_radiomap`](../../indoorloc/methods/gaussian_process.py)<br>`GPRadioMapLocalizer` | Gaussian-process radio map with maximum-likelihood localization (Ferris et al., 2006). | numpy only | no (fill first) | Ferris, B., Hähnel, D., Fox, D., "Gaussian processes for signal strength-based location estimation", Robotics: Science and Systems II, 2006. [doi:10.15607/RSS.2006.II.039](https://doi.org/10.15607/RSS.2006.II.039) (+1 more in the docstring) |
| [`hierarchical`](../../indoorloc/methods/hierarchical.py)<br>`HierarchicalLocalizer` | Coarse-to-fine localization: building -> floor -> position, one sub-model per group. | numpy only | as its members | Marques, N., Meneses, F., Moreira, A., "Combining similarity functions and majority rules for multi-building, multi-floor, WiFi positioning", IPIN 2012. [doi:10.1109/IPIN.2012.6418937](https://doi.org/10.1109/IPIN.2012.6418937) (+1 more in the docstring) |
| [`horus`](../../indoorloc/methods/probabilistic.py)<br>`HorusLocalizer` | Horus: maximum-likelihood fingerprinting with per-location, per-AP Gaussian RSSI models. | numpy only | yes | Youssef, M., Agrawala, A., "The Horus WLAN location determination system", MobiSys 2005. [doi:10.1145/1067170.1067193](https://doi.org/10.1145/1067170.1067193) (+1 more in the docstring) |
| [`knn`](../../indoorloc/methods/neighbors.py)<br>`KNNLocalizer` | k-NN fingerprinting: average the positions of the `k` nearest reference scans. | numpy only | no (fill first) | P. Bahl, V. N. Padmanabhan, "RADAR: an in-building RF-based user location and tracking system", IEEE INFOCOM 2000. [doi:10.1109/INFCOM.2000.832252](https://doi.org/10.1109/INFCOM.2000.832252). (+1 more in the docstring) |
| [`rf`](../../indoorloc/methods/sklearn_wrap.py)<br>alias: `random_forest`<br>`RandomForestLocalizer` | Random-forest fingerprinting: one multi-output forest for the position, forests for labels. | `[sklearn]` | no (fill first) | Breiman, L., "Random forests", Machine Learning 45(1), 2001. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) (+1 more in the docstring) |
| [`stacking`](../../indoorloc/methods/ensemble.py)<br>`StackingLocalizer` | Stacked generalization: a meta-learner fitted on out-of-fold member predictions. | numpy only | as its members | Wolpert, D. H., "Stacked generalization", Neural Networks 5(2), 1992. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1) (+1 more in the docstring) |
| [`svm`](../../indoorloc/methods/sklearn_wrap.py)<br>`SVMLocalizer` | Support-vector fingerprinting: one epsilon-SVR per coordinate axis, SVC for labels. | `[sklearn]` | no (fill first) | Smola, A. J., Schölkopf, B., "A tutorial on support vector regression", Statistics and Computing 14(3), 2004. [doi:10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88) (+2 more in the docstring) |
| [`wknn`](../../indoorloc/methods/neighbors.py)<br>alias: `weighted_knn`<br>`WKNNLocalizer` | Weighted k-NN: KNNLocalizer with inverse-distance weights by default. | numpy only | no (fill first) | — |

**Model-based**

| Name | What it does | Extra | NaN input | Reference |
| --- | --- | --- | --- | --- |
| [`aoa`](../../indoorloc/methods/aoa.py)<br>`AoALocalizer` | 2-D position from angles of arrival at anchors with known positions and orientations. | numpy only | yes | R. O. Schmidt, "Multiple emitter location and signal parameter estimation", IEEE Transactions on Antennas and Propagation 34(3), 1986. [doi:10.1109/TAP.1986.1143830](https://doi.org/10.1109/TAP.1986.1143830). (+4 more in the docstring) |
| [`centroid`](../../indoorloc/methods/geometric.py)<br>`WeightedCentroidLocalizer` | Weighted centroid of the anchors that hear the target. | numpy only | yes | J. Blumenthal, R. Grossmann, F. Golatowski, D. Timmermann, "Weighted centroid localization in Zigbee-based sensor networks", IEEE International Symposium on Intelligent Signal Processing (WISP), 2007. [doi:10.1109/WISP.2007.4447528](https://doi.org/10.1109/WISP.2007.4447528). (+1 more in the docstring) |
| [`pathloss`](../../indoorloc/methods/pathloss.py)<br>`PathLossLocalizer` | Maximum-likelihood localization with a log-distance path-loss model per anchor. | numpy only | yes | T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice Hall, 2002. ISBN 0-13-042232-0. (+4 more in the docstring) |
| [`tdoa`](../../indoorloc/methods/geometric.py)<br>`TDOALocalizer` | Position from range differences (TDoA) with Chan and Ho's closed form. | numpy only | yes | Y. T. Chan, K. C. Ho, "A simple and efficient estimator for hyperbolic location", IEEE Transactions on Signal Processing 42(8), 1994. [doi:10.1109/78.301830](https://doi.org/10.1109/78.301830). (+1 more in the docstring) |
| [`trilateration`](../../indoorloc/methods/geometric.py)<br>alias: `multilateration`<br>`TrilaterationLocalizer` | Position from ranges to anchors at known positions (ToA, two-way RTT, UWB). | numpy only | yes | W. H. Foy, "Position-location solutions by Taylor-series estimation", IEEE Transactions on Aerospace and Electronic Systems AES-12(2), 1976. [doi:10.1109/TAES.1976.308294](https://doi.org/10.1109/TAES.1976.308294). (+4 more in the docstring) |
| [`vlc`](../../indoorloc/methods/vlc.py)<br>`LambertianLocalizer` | Visible light positioning by received-power ranging or model fitting (Lambertian LOS). | numpy only | yes | Y. Zhuang, L. Hua, L. Qi, J. Yang, P. Cao, Y. Cao, Y. Wu, J. Thompson, H. Haas, "A survey of positioning systems using visible LED lights", IEEE Communications Surveys & Tutorials 20(3):1963-1988, 2018. [doi:10.1109/COMST.2018.2806558](https://doi.org/10.1109/COMST.2018.2806558). (+3 more in the docstring) |

**Sequence matching**

| Name | What it does | Extra | NaN input | Reference |
| --- | --- | --- | --- | --- |
| [`magnetic_dtw`](../../indoorloc/methods/magnetic.py)<br>`MagneticDTWLocalizer` | Localization by subsequence DTW of the recent magnetic sequence against reference walks. | numpy only | no | K. P. Subbu, B. Gozick, R. Dantu, "LocateMe: magnetic-fields-based indoor localization using smartphones", ACM Transactions on Intelligent Systems and Technology 4(4), 2013. [doi:10.1145/2508037.2508054](https://doi.org/10.1145/2508037.2508054). (+3 more in the docstring) |

**Deep learning**

| Name | What it does | Extra | NaN input | Reference |
| --- | --- | --- | --- | --- |
| [`cnn1d`](../../indoorloc/methods/deep/localizer.py)<br>`CNN1DLocalizer` | 1-D convolutional fingerprinting: `DeepLocalizer` with the `"cnn1d"` backbone. | `[deep]` | no (fill first) | X. Song, X. Fan, C. Xiang, Q. Ye, L. Liu, Z. Wang, X. He, N. Yang and G. Fang, "A Novel Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting", IEEE Access 7:110698-110709, 2019. <https://doi.org/10.1109/ACCESS.2019.2933921> |
| [`deep`](../../indoorloc/methods/deep/localizer.py)<br>`DeepLocalizer` | Deep fingerprinting: a backbone network with multi-task heads, trained end to end. | `[deep]` | no (fill first) | K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018. <https://doi.org/10.1186/s41044-018-0031-2> (+2 more in the docstring) |
| [`mlp`](../../indoorloc/methods/deep/localizer.py)<br>`MLPLocalizer` | Multi-layer perceptron fingerprinting: `DeepLocalizer` with the `"mlp"` backbone. | `[deep]` | no (fill first) | K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018. <https://doi.org/10.1186/s41044-018-0031-2> |
<!-- /catalog:methods -->

Domain adaptation (`indoorloc.methods.transfer`) consists of L2-style transforms that learn from
labelled source scans and unlabelled target scans:

<!-- catalog:transfer -->
| Name | Kind | What it does | Extra | Reference |
| --- | --- | --- | --- | --- |
| [`CORAL`](../../indoorloc/methods/transfer.py) | class | CORrelation ALignment: re-colour the source features with the target covariance. | numpy only | B. Sun, J. Feng and K. Saenko, "Return of Frustratingly Easy Domain Adaptation", Proceedings of the AAAI Conference on Artificial Intelligence 30(1), 2016. <https://doi.org/10.1609/aaai.v30i1.10306> |
| [`TCA`](../../indoorloc/methods/transfer.py) | class | Transfer Component Analysis: a shared low-dimensional embedding in which the domains match. | numpy only | S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via Transfer Component Analysis", IEEE Transactions on Neural Networks 22(2):199-210, 2011. <https://doi.org/10.1109/TNN.2010.2091281> (its experiments include cross-domain WiFi localization) |
| [`SkadaAdapter`](../../indoorloc/methods/transfer.py) | class | Any feature-level adapter of the skada library as an L2 transform (`fit(X, target=...)`). | `[transfer]` | skada, "Scikit-learn-compatible domain adaptation", <https://github.com/scikit-adaptation/skada>; (+1 more in the docstring) |
| [`mmd`](../../indoorloc/methods/transfer.py) | function | Squared maximum mean discrepancy between two samples, `MMD^2(X, Y)`. | numpy only | A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schoelkopf and A. Smola, "A Kernel Two-Sample Test", Journal of Machine Learning Research 13:723-773, 2012. <https://jmlr.org/papers/v13/gretton12a.html> |
<!-- /catalog:transfer -->

## Examples by family

All numbers below come from the simulated office; they illustrate the calls, not how the
methods compare in real buildings. For real data, see [docs/benchmarks.md](../benchmarks.md).

### Fingerprinting

```python
for name, params in [("horus", {}), ("gp_radiomap", {}),
                     ("ensemble", {"localizers": ["wknn", "horus", "gp_radiomap"], "combine": "median"})]:
    m = iloc.create_model(name, preprocess=FillMissing(-104), **params).fit(train)
    print(name, round(m.evaluate(test).mean_error, 3))
# horus 1.705
# gp_radiomap 1.989
# ensemble 1.704
```

`svm`, `rf`, `extratrees` and `gbdt` need scikit-learn (`[sklearn]`); `mlp`, `cnn1d` and `deep`
need torch (`[deep]`, and timm for `deep` with a timm backbone):

```python
# requires: torch
mlp = iloc.create_model("mlp", hidden=(128, 64), epochs=50, random_state=0,
                        preprocess=FillMissing(-104)).fit(train)
print(round(mlp.evaluate(test).mean_error, 3))
# 2.41
```

The MLP result is deterministic for a given `random_state` and torch thread count.

### Model-based

The anchor geometry comes from the table's `meta["anchors"]` (for the simulator) or from the
deployment plan. `from_meta` reads it from a table, and for AoA also the array orientations in
`meta["anchor_orientations"]`:

```python
for modality, cls in [("ranges", iloc.TrilaterationLocalizer), ("tdoa", iloc.TDOALocalizer),
                      ("aoa", iloc.AoALocalizer)]:
    tr, te = iloc.load_dataset("synthetic_office", modality=modality)
    m = cls.from_meta(te.meta)                   # no fit needed: the geometry is the model
    print(modality, te.X.shape, round(m.evaluate(te).mean_error, 3))
# ranges (200, 4) 0.603
# tdoa (200, 4) 0.48
# aoa (200, 4) 3.545
```

The geometry is all these models need, so `localize` works before `fit`, and `fit(X)` without
positions only records the input size. The exception is `calibrate=True`: `fit(X, positions)`
(or `fit(table)`) then learns a per-anchor bias and the noise scale from labelled measurements,
and positions are required. Path loss learns its noise scale, and any parameter left unset,
from labelled scans, so its `fit` always needs positions.

AoA angles are measured in each array's own frame. Built by hand without the orientations
(`create_model("aoa", anchors=te.meta["anchors"])`), the same test set gives a mean error of
74.9 m instead of 3.5 m. When `fit` gets positions, it warns if an array's angles disagree with
them by a median of more than 45 degrees.

3-D anchors that lie in (nearly) one plane, such as UWB anchors on a ceiling, cannot tell a tag
below the plane from its mirror image above it. In the 3-D simulated office the four ranging
anchors lie in one plane, so the 3-D solve places no sample. When the tag height is known,
`height=` solves for `(x, y)` only and returns `(x, y, height)`:

```python
tr3, te3 = iloc.load_dataset("synthetic_office", modality="ranges", dim=3)
for height in (None, 1.2):
    print(height, iloc.TrilaterationLocalizer.from_meta(te3.meta, height=height).evaluate(te3))
# None no sample placed  floor n/a  building n/a  (n=200, 200 not placed)
# 1.2 mean 0.6062  median 0.4958  P90 1.2541  floor n/a  building n/a  (n=200)
```

Path loss and weighted centroid localize from RSSI and known access-point positions. With
`anchors=None`, the path-loss model also estimates the transmitter positions from the labelled
training scans.

```python
anchors = train.meta["anchors"]
for name in ("pathloss", "centroid"):
    m = iloc.create_model(name, anchors=anchors).fit(train)
    print(name, round(m.evaluate(test).mean_error, 3))
# pathloss 3.068
# centroid 3.645
```

Visible-light positioning takes the LED layout and optical constants from the table:

```python
from indoorloc.methods.vlc import LambertianLocalizer

vlc_train, vlc_test = iloc.load_dataset("synthetic_office", modality="vlc")
res = LambertianLocalizer.from_meta(vlc_test.meta).fit(vlc_train).evaluate(vlc_test)
print(vlc_test.X.shape, round(res.mean_error, 4), res.n_failed)
# (200, 176) 0.0207 3
```

`n_failed` counts the samples a method could not place (here, points where the visible LEDs
leave a mirror ambiguity). They are left out of the error statistics, so report the count next
to the errors.

### Sequence matching

`magnetic_dtw` matches the recent window of magnetic features against reference walks by
subsequence dynamic time warping. Walk boundaries are passed to `fit`, and `localize_walks` keeps
query windows inside each walk:

```python
walks = iloc.load_dataset("synthetic_office", split="trajectory", modality="magnetic", n_trajectories=12)
traj = walks.groups["trajectory"]
reference, query = walks[traj < 10], walks[traj >= 10]
dtw = iloc.create_model("magnetic_dtw", window=30).fit(reference, trajectory=reference.groups["trajectory"])
print(iloc.evaluate(query, dtw.localize_walks(query)))
# mean 6.7558  median 1.2264  P90 17.8236  floor 100.00 %  building n/a  (n=1200, 58 not placed)
```

The 58 rows that are not placed are the first samples of each query walk, before a full window
is available. On measured data (the ILC 2020 sample) magnetic matching alone was far weaker; the
class docstring gives those numbers.

### Domain adaptation

CORAL re-colours the source scans with the covariance of unlabelled target scans. Here the
"new device" is the simulated test set with a gain and an offset applied:

```python
from indoorloc.methods import LocalizerPipeline
from indoorloc.methods.transfer import CORAL

fill = FillMissing(-104)
X_src, X_new = fill.transform(train.X), fill.transform(0.8 * test.X - 12.0)    # simulated device shift
plain = iloc.create_model("wknn").fit(X_src, train.pos)
adapted = LocalizerPipeline(CORAL(align_mean=True), iloc.create_model("wknn"))
adapted.fit(X_src, train.pos, preprocess__target=X_new)
print(round(iloc.evaluate(test.pos, plain.predict(X_new)).mean_error, 3),
      round(iloc.evaluate(test.pos, adapted.predict(X_new)).mean_error, 3))
# 2.441 1.934
```
