"""L3 transfer: CORAL, TCA, mmd and the skada bridge, checked against closed forms and known limits."""
from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

from conftest import HEAVY, PROJECT, UJI_ROOT
from indoorloc.core import NotFittedError, SampleTable, load_model
from indoorloc.methods import LocalizerPipeline
from indoorloc.methods.neighbors import KNNLocalizer
from indoorloc.methods.transfer import CORAL, TCA, SkadaAdapter, mmd


@pytest.fixture(autouse=True, scope="module")
def _few_blas_threads():
    """Small matrices: dozens of BLAS threads only spin (and the machine may be shared)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=2, user_api="blas"):
        yield


def _two_domains(seed=0, n=300, shift=(4.0, 0.0)):
    """Pan et al.'s 2-D toy setting: the domains differ along x1, which also has the larger pooled
    variance, so PCA keeps the domain difference and TCA must drop it."""
    rng = np.random.default_rng(seed)
    scale = np.array([1.0, np.sqrt(2.0)])
    return rng.normal(size=(n, 2)) * scale, rng.normal(size=(n, 2)) * scale + np.asarray(shift)


def _brute_mmd(X, Y, k, unbiased):
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    if unbiased:
        return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                - 2 * kxy.mean())
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


# ------------------------------------------------------------------------------------------------ mmd
def test_mmd_matches_closed_forms_and_brute_force():
    assert mmd([[0.0]], [[1.0]], gamma=1.0) == pytest.approx(2 - 2 * np.exp(-1), abs=1e-15)
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(40, 3)), rng.normal(size=(30, 3)) + 0.5
    linear = lambda A, B: A @ B.T  # noqa: E731
    rbf = lambda A, B: np.exp(-0.3 * ((A[:, None] - B[None]) ** 2).sum(-1))  # noqa: E731
    assert mmd(X, Y, kernel="linear") == pytest.approx(np.sum((X.mean(0) - Y.mean(0)) ** 2), rel=1e-12)
    for unbiased in (False, True):
        exact_linear = _brute_mmd(X, Y, linear, unbiased)
        assert mmd(X, Y, kernel="linear", unbiased=unbiased) == pytest.approx(exact_linear, rel=1e-10)
        exact = _brute_mmd(X, Y, rbf, unbiased)
        assert mmd(X, Y, gamma=0.3, unbiased=unbiased) == pytest.approx(exact, rel=1e-10)
        assert mmd(X, Y, gamma=0.3, unbiased=unbiased, chunk_size=7) == pytest.approx(exact, rel=1e-12)
    assert mmd(X, X, gamma=0.3) == pytest.approx(0.0, abs=1e-12)
    d2 = ((np.concatenate([X, Y])[:, None] - np.concatenate([X, Y])[None]) ** 2).sum(-1)
    median_gamma = 1 / (2 * np.median(d2[np.triu_indices(70, 1)]))  # Gretton et al.'s median heuristic
    assert mmd(X, Y) == pytest.approx(mmd(X, Y, gamma=median_gamma), rel=1e-12)
    with pytest.raises(ValueError, match="FillMissing"):
        mmd(np.array([[np.nan]]), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="features"):
        mmd(X, Y[:, :2])
    with pytest.raises(ValueError, match="gamma"):
        mmd(X, Y, gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        TCA(kernel="rbf", gamma=-1.0).fit(X, target=Y)
    with pytest.raises(ValueError, match="chunk_size"):
        mmd(X, Y, chunk_size=0)


# ---------------------------------------------------------------------------------------------- CORAL
def test_coral_recolours_the_source_with_the_target_covariance():
    rng = np.random.default_rng(2)
    Xs = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4)) - 70
    Xt = rng.normal(size=(400, 4)) @ rng.normal(size=(4, 4)) - 60
    coral = CORAL(reg=0.0).fit(Xs, target=Xt)
    Z = coral.transform_source(Xs)
    assert np.allclose(np.cov(Z, rowvar=False), np.cov(Xt, rowvar=False), rtol=1e-9, atol=1e-9)  # reg = 0: exact
    assert np.allclose(Z.mean(0), Xs.mean(0), atol=1e-9)  # second order only; the mean stays
    aligned = CORAL(reg=0.0, align_mean=True).fit(Xs, target=Xt).transform_source(Xs)
    assert np.allclose(aligned.mean(0), Xt.mean(0), atol=1e-9)
    assert np.array_equal(coral.transform(Xt), Xt) and coral.transform(Xt).dtype == np.float64  # target: as is
    assert np.array_equal(coral.fit_transform(Xs, target=Xt), Z)


def test_coral_one_feature_and_paper_algorithm():
    rng = np.random.default_rng(3)
    xs, xt = rng.normal(2.0, 3.0, size=(200, 1)), rng.normal(-1.0, 0.5, size=(150, 1))
    reg = 1.0
    factor = np.sqrt((xt.var(ddof=1) + reg) / (xs.var(ddof=1) + reg))  # closed form in one dimension
    got = CORAL(reg=reg).fit(xs, target=xt).transform_source(xs)
    assert np.allclose(got, (xs - xs.mean()) * factor + xs.mean(), rtol=1e-12)
    # Zero-mean features: identical to Algorithm 1 of Sun et al. (2016), Ds * Cs^(-1/2) * Ct^(1/2)
    sqrtm = pytest.importorskip("scipy.linalg").sqrtm
    Xs = rng.normal(size=(300, 3)) @ rng.normal(size=(3, 3))
    Xs -= Xs.mean(0)
    Xt = rng.normal(size=(250, 3)) @ rng.normal(size=(3, 3))
    cs, ct = np.cov(Xs, rowvar=False) + np.eye(3), np.cov(Xt, rowvar=False) + np.eye(3)
    paper = Xs @ np.linalg.inv(np.real(sqrtm(cs))) @ np.real(sqrtm(ct))
    assert np.allclose(CORAL().fit(Xs, target=Xt).transform_source(Xs), paper, rtol=1e-8, atol=1e-10)


def test_coral_undoes_a_device_gain_and_offset():
    """A target phone that reads a * RSSI + b: CORAL(align_mean) maps the survey exactly onto it, so a
    1-NN trained on the adapted survey localizes the new phone's scans without error."""
    rng = np.random.default_rng(4)
    aps, points = rng.uniform(0, 50, (12, 2)), rng.uniform(0, 50, (200, 2))
    survey = -40 - 25 * np.log10(1 + np.linalg.norm(points[:, None] - aps[None], axis=-1))
    phone = 0.8 * survey - 12.0  # lower gain, 12 dB offset
    plain = KNNLocalizer(k=1).fit(survey, points)
    adapted = LocalizerPipeline(CORAL(reg=0.0, align_mean=True), KNNLocalizer(k=1))
    adapted.fit(survey, points, preprocess__target=phone)  # unlabelled scans of the new phone
    assert np.allclose(adapted.preprocess_.transform_source(survey), phone, atol=1e-8)
    assert np.allclose(adapted.predict(phone), points, atol=1e-9)
    assert np.linalg.norm(plain.predict(phone) - points, axis=1).mean() > 5.0  # without adaptation: metres off


def test_coral_input_rules():
    X = np.zeros((5, 2)) + np.arange(5)[:, None] * [1.0, 2.0]
    with pytest.raises(NotFittedError):
        CORAL().transform(X)
    with pytest.raises(ValueError, match="target"):
        CORAL().fit(X)
    with pytest.raises(ValueError, match="FillMissing"):
        CORAL().fit(np.where(X > 3, np.nan, X), target=X)
    with pytest.raises(ValueError, match="singular"):
        CORAL(reg=0.0).fit(X, target=X)  # collinear features
    with pytest.raises(ValueError, match="features"):
        CORAL().fit(X, target=X[:, :1])
    table = SampleTable(X, np.zeros((5, 2)), floor=np.arange(5))
    out = CORAL().fit_transform(table, target=X + 1)
    assert isinstance(out, SampleTable) and out.floor.tolist() == list(range(5))


# ------------------------------------------------------------------------------------------------ TCA
def test_tca_drops_the_domain_direction_that_pca_keeps():
    Xs, Xt = _two_domains()
    tca = TCA(n_components=1, mu=1.0).fit(Xs, target=Xt)
    zs, zt = tca.transform(Xs), tca.transform(Xt)
    assert abs(tca.components_[1, 0]) > 0.999  # TCA keeps x2 ...
    assert mmd(zs, zt, kernel="linear") < 1e-9 * np.var(np.concatenate([zs, zt]))  # ... where the domains coincide
    pca_like = TCA(n_components=1, mu=1e12).fit(Xs, target=Xt)
    assert abs(pca_like.components_[0, 0]) > 0.99  # mu -> inf: the direction of largest pooled variance (x1)


def test_tca_with_large_mu_is_pca():
    Xs, Xt = _two_domains(seed=5, shift=(1.0, 3.0))
    X = np.concatenate([Xs, Xt])
    _, V = np.linalg.eigh(np.cov(X, rowvar=False))
    V = V[:, ::-1]
    tca = TCA(n_components=2, mu=1e12, max_samples=None).fit(Xs, target=Xt)
    signs = np.sign(np.sum(tca.components_ * V, axis=0))
    assert np.allclose(tca.components_, V * signs, atol=1e-8)  # unit-length principal axes
    assert np.allclose(tca.transform(X), (X - X.mean(0)) @ (V * signs), atol=1e-6)  # PCA scores
    assert np.all(np.diff(tca.eigenvalues_) <= 0)


def test_tca_rbf_embedding_matches_the_domains_and_is_consistent():
    Xs, Xt = _two_domains(seed=6)
    tca = TCA(n_components=2, kernel="rbf", mu=1.0).fit(Xs, target=Xt)
    zs, zt = tca.transform(Xs), tca.transform(Xt)
    assert mmd(zs, zt) < mmd(Xs, Xt) / 20
    assert np.allclose(tca.fit_transform(Xs, target=Xt), zs, atol=1e-12)  # fit_transform = fit, then transform
    assert np.allclose(tca.transform(Xs[3]), zs[3], atol=1e-12)  # one scan (F,) as a batch row
    table = tca.transform(SampleTable(Xt, np.zeros((len(Xt), 2)), meta={"feature_names": ("a", "b")}))
    assert table.X.shape == (len(Xt), 2) and table.meta["feature_names"] == ("tca0", "tca1")
    with pytest.raises(ValueError, match="rank"):
        TCA(n_components=3).fit(Xs, target=Xt)  # a 2-D linear kernel has rank 2


def _centred_rbf(X, gamma):
    K = np.exp(-gamma * ((X[:, None] - X[None]) ** 2).sum(-1))
    H = np.eye(len(X)) - 1.0 / len(X)
    return K, H @ K @ H


def test_tca_rbf_with_large_mu_is_kernel_pca():
    """mu -> inf: the RBF embedding is kernel PCA (Schoelkopf et al., 1998) on the pooled samples, new rows
    centred like sklearn's KernelCenterer; also computed in row blocks without changing a bit."""
    Xs, Xt = _two_domains(seed=10, n=120)
    X, X_new, g = np.concatenate([Xs, Xt]), np.random.default_rng(11).normal(size=(30, 2)) * 3, 0.2
    tca = TCA(n_components=3, kernel="rbf", gamma=g, mu=1e12, max_samples=None).fit(Xs, target=Xt)
    K, Kc = _centred_rbf(X, g)
    lam, V = np.linalg.eigh(Kc)
    lam, V = lam[::-1][:3], V[:, ::-1][:, :3]
    K_new = np.exp(-g * ((X_new[:, None] - X[None]) ** 2).sum(-1))
    K_new_c = K_new - K.mean(0)[None] - K_new.mean(1, keepdims=True) + K.mean()
    ref = K_new_c @ (V / np.sqrt(lam))  # unit-length feature-space directions
    got = tca.transform(X_new)
    signs = np.sign(np.sum(ref * got, axis=0))
    assert np.allclose(got, ref * signs, atol=1e-10)
    assert np.allclose(tca.transform(X), V * np.sqrt(lam) * signs, atol=1e-10)  # fitted rows: the kPCA scores
    tca._block = 7  # several blocks, the last one partial
    assert np.array_equal(tca.transform(X_new), got)


def test_tca_solves_the_generalized_eigenproblem_for_any_mu():
    """The exact B^(-1/2) route (B = mu I + v v^T, ridge plus rank one) must solve Pan et al.'s
    K H K w = lambda (K L K + mu I) w with the largest lambdas, also when mu is comparable to |v|^2."""
    Xs, Xt = _two_domains(seed=12, n=100, shift=(1.5, 0.0))
    X, g = np.concatenate([Xs, Xt]), 0.2
    _, Kc = _centred_rbf(X, g)
    e = np.r_[np.full(len(Xs), 1 / len(Xs)), np.full(len(Xt), -1 / len(Xt))]
    v = Kc @ e
    A = Kc @ Kc
    for mu in (1e-3, 1.0, float(v @ v), 100.0):
        tca = TCA(n_components=4, kernel="rbf", gamma=g, mu=mu, max_samples=None).fit(Xs, target=Xt)
        W, lam = tca.dual_coef_, tca.eigenvalues_
        B = np.outer(v, v) + mu * np.eye(len(X))
        assert np.abs(A @ W - (B @ W) * lam).max() <= 1e-9 * np.abs(A @ W).max()
        top = np.sort(np.linalg.eigvals(np.linalg.solve(B, A)).real)[::-1][:4]
        assert np.allclose(lam, top, rtol=1e-8)
        assert np.allclose(np.einsum("ij,ij->j", W, Kc @ W), 1.0)  # unit feature-space length (documented)


def test_tca_subsampling_is_seeded():
    Xs, Xt = _two_domains(seed=7, n=500)
    a = TCA(n_components=2, kernel="rbf", max_samples=100, random_state=1).fit(Xs, target=Xt)
    b = TCA(n_components=2, kernel="rbf", max_samples=100, random_state=1).fit(Xs, target=Xt)
    c = TCA(n_components=2, kernel="rbf", max_samples=100, random_state=2).fit(Xs, target=Xt)
    assert (a.n_source_fit_, a.n_target_fit_) == (100, 100) and np.array_equal(a.X_fit_, b.X_fit_)
    assert np.array_equal(a.transform(Xt), b.transform(Xt)) and not np.array_equal(a.X_fit_, c.X_fit_)


# -------------------------------------------------------------------------------- pipeline and saving
@pytest.mark.parametrize("adapter", [CORAL(align_mean=True), TCA(n_components=3), TCA(n_components=3, kernel="rbf")])
def test_adapters_in_a_pipeline_save_as_arrays(adapter, tmp_path):
    rng = np.random.default_rng(8)
    X, pos, Xt = rng.normal(-70, 8, (80, 6)), rng.uniform(0, 20, (80, 2)), rng.normal(-75, 6, (50, 6))
    model = LocalizerPipeline(adapter, KNNLocalizer(k=3)).fit(X, pos, floor=np.arange(80) % 2, preprocess__target=Xt)
    pred = model.localize(Xt)
    loaded = load_model(model.save(tmp_path / "m"))
    again = loaded.localize(Xt)
    assert np.array_equal(again.pos, pred.pos) and np.array_equal(again.floor, pred.floor)
    assert not any(f.suffix in (".pkl", ".pickle") for f in (tmp_path / "m").iterdir())


# ---------------------------------------------------------------------------------------------- skada
class _FakeCORALAdapter:  # stands in for skada.CORALAdapter: records the calls it receives
    def __init__(self, reg=1.0):
        self.reg = reg

    def fit_transform(self, X, y=None, *, sample_domain=None):
        self.domains_ = np.unique(sample_domain)
        self.n_source_ = int(np.sum(sample_domain > 0))
        self.shift_ = X[sample_domain < 0].mean(0) - X[sample_domain > 0].mean(0)
        return np.where((sample_domain > 0)[:, None], X + self.shift_, X)

    def transform(self, X, sample_domain=None):
        self.transform_domains_ = np.unique(sample_domain)
        return X.copy()


class _FakeReweight(_FakeCORALAdapter):
    def fit_transform(self, X, y=None, *, sample_domain=None):
        return {"X": X, "sample_weight": np.ones(len(X))}


def test_skada_adapter_follows_the_sample_domain_convention(monkeypatch, tmp_path):
    """Checked against a stand-in with skada's adapter API; the real package is not installed here."""
    monkeypatch.setitem(sys.modules, "skada", types.SimpleNamespace(CORALAdapter=_FakeCORALAdapter,
                                                                    KMMReweightAdapter=_FakeReweight))
    Xs, Xt = np.zeros((4, 2)), np.ones((3, 2))
    step = SkadaAdapter("CORALAdapter", params={"reg": 0.5})
    out = step.fit_transform(Xs, target=Xt)
    assert step.adapter_.reg == 0.5 and step.adapter_.domains_.tolist() == [-2, 1] and step.adapter_.n_source_ == 4
    assert out.shape == (4, 2) and np.allclose(out, 1.0)  # only the source rows come back, adapted
    assert np.array_equal(step.transform(Xt), Xt) and step.adapter_.transform_domains_.tolist() == [-2]
    with pytest.raises(TypeError, match="reweighting"):
        SkadaAdapter("KMMReweightAdapter").fit(Xs, target=Xt)
    with pytest.raises(ValueError, match="no adapter"):
        SkadaAdapter("Nope").fit(Xs, target=Xt)
    with pytest.raises(TypeError, match="SkadaAdapter cannot be saved"):  # never pickled, and says why
        step.save(tmp_path / "skada")


def test_skada_coral_equals_ours_under_its_conventions():
    """Real skada (skipped when absent): its CORALAdapter centres both domains, uses ddof=0 covariances
    and, with reg=None, no shrinkage. With those conventions it equals CORAL(reg=0, align_mean=True)."""
    pytest.importorskip("skada")
    rng = np.random.default_rng(9)
    Xs = rng.normal(size=(200, 3)) @ rng.normal(size=(3, 3))
    Xt = rng.normal(size=(150, 3)) @ rng.normal(size=(3, 3)) + 1
    step = SkadaAdapter("CORALAdapter", params={"reg": None})
    zs, zt = step.fit_transform(Xs, target=Xt), step.transform(Xt)
    ours = CORAL(reg=0.0, align_mean=True).fit(Xs, target=Xt)
    ddof = np.sqrt(len(Xs) / (len(Xs) - 1) * (len(Xt) - 1) / len(Xt))
    assert np.allclose(zt, Xt - Xt.mean(0), atol=1e-12)
    assert np.allclose(zs, ddof * (ours.transform_source(Xs) - Xt.mean(0)), atol=1e-10)


def test_skada_missing_names_the_extra():
    code = (f"import sys\nfor m in {HEAVY!r}: sys.modules[m] = None\n"
            "import numpy as np\nfrom indoorloc.methods.transfer import SkadaAdapter\n"
            "try:\n    SkadaAdapter().fit(np.zeros((3, 2)), target=np.ones((3, 2)))\n"
            "except ImportError as e:\n    assert \"indoorloc[transfer]\" in str(e), e\n"
            "else:\n    raise AssertionError('expected ImportError')")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------------------------------- real data
@pytest.mark.skipif(not (UJI_ROOT / "validationData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_ujiindoorloc_phones_become_closer_after_adaptation():
    """UJIIndoorLoc validation split: phones 13 and 14 (also in the training survey) against the nine
    phones that appear only in validation. Both adapters must shrink the MMD between the groups."""
    from indoorloc.datasets import load_dataset
    from indoorloc.signals import FillMissing

    t = FillMissing(-104).transform(load_dataset("ujiindoorloc", split="test", root=UJI_ROOT, download=False))
    seen = np.isin(t.groups["device"], [13, 14])
    Xs, Xt = t.X[seen].astype(np.float64), t.X[~seen].astype(np.float64)
    assert (len(Xs), len(Xt)) == (397, 714)
    before = mmd(Xs, Xt, gamma=1e-4)
    coral = CORAL(align_mean=True).fit(Xs, target=Xt)
    assert mmd(coral.transform_source(Xs), Xt, gamma=1e-4) < before
    assert mmd(coral.transform_source(Xs), Xt, kernel="linear") < 1e-12 * mmd(Xs, Xt, kernel="linear")
    tca = TCA(n_components=20).fit(Xs, target=Xt)
    zs, zt = tca.transform(Xs), tca.transform(Xt)
    assert mmd(zs, zt, kernel="linear") < 1e-6 * np.var(np.concatenate([zs, zt]), axis=0).sum()
