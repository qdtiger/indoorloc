"""L3 deep: DeepLocalizer / MLPLocalizer / CNN1DLocalizer on CPU, with known-result checks."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import HEAVY, PROJECT
from indoorloc.core import clone, load_model
from indoorloc.methods import LocalizerPipeline, create_model
from indoorloc.methods.deep import CNN1DLocalizer, DeepLocalizer, MLPLocalizer
from indoorloc.signals import FillMissing

torch = pytest.importorskip("torch")

UJI_ORIGIN = np.array([-7600.0, 4864900.0])  # UJIIndoorLoc-sized coordinates (EPSG:3857 metres)


@pytest.fixture(autouse=True, scope="module")
def _two_threads():
    """Small nets: extra torch/BLAS threads only spin (and the machine may be shared)."""
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
    else:
        with threadpool_limits(limits=2, user_api="blas"):
            yield
    finally:
        torch.set_num_threads(before)


def _clusters(seed=0, n=240, n_features=12):
    """Scans around 4 reference points; floor (with a basement, -1) and building follow the point."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (4, n_features))
    which = np.arange(n) % 4
    X = (centres[which] + rng.normal(0, 0.3, (n, n_features))).astype(np.float32)
    pos = UJI_ORIGIN + np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 30.0], [40.0, 30.0]])[which]
    return X, pos, np.array([-1, 0, 3, 3])[which], np.array([0, 0, 1, 1])[which]


def test_a_linear_network_recovers_least_squares():
    """hidden=() makes the network an affine map of the scaled input; trained full batch it must reach
    the least-squares solution, also at UJIIndoorLoc's 4.86e6 m northings (0.1 did not scale targets)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = X.astype(np.float64) @ rng.normal(0, 10, (6, 2)) + UJI_ORIGIN
    model = DeepLocalizer(hidden=(), dropout=0.0, batch_norm=False, weight_decay=0.0, epochs=300,
                          batch_size=300, lr=0.05, patience=None).fit(X, y)
    A = np.c_[X.astype(np.float64), np.ones(len(X))]
    exact = A @ np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.abs(model.predict(X) - exact).max() < 1e-3  # metres, for coordinates ~5e6 m
    assert np.array_equal(model.pos_mean_, y.mean(0)) and model.pos_mean_.dtype == np.float64
    assert model.pos_scale_ == pytest.approx(np.sqrt(y.var(0).mean()), rel=1e-15)  # one isotropic scale
    assert model.x_mean_ == pytest.approx(X.astype(np.float64).mean(), rel=1e-12)


def test_multitask_heads_predict_floor_and_building_labels():
    X, pos, floor, building = _clusters()
    model = MLPLocalizer(hidden=(32,), dropout=0.0, epochs=40, batch_size=32, patience=None)
    p = model.fit(X, pos, floor=floor, building=building).localize(X)
    assert model.floor_classes_.tolist() == [-1, 0, 3] and model.building_classes_.tolist() == [0, 1]
    assert np.array_equal(p.floor, floor) and np.array_equal(p.building, building)  # negative floors are floors
    assert np.linalg.norm(p.pos - pos, axis=1).mean() < 3.0  # reference points 30-50 m apart
    unlabelled = MLPLocalizer(hidden=(8,), epochs=2, patience=None).fit(X, pos).localize(X[:3])
    assert unlabelled.floor is None and unlabelled.building is None
    assert unlabelled.pos.dtype == np.float64 and unlabelled.pos.shape == (3, 2)


def test_training_is_seeded_and_leaves_the_caller_rng_alone():
    X, pos, floor, _ = _clusters(seed=2)
    make = lambda seed: MLPLocalizer(hidden=(16,), epochs=5, patience=2, random_state=seed)  # noqa: E731
    torch.manual_seed(123)
    np.random.seed(7)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()[1].copy()
    a = make(0).fit(X, pos, floor=floor)
    b = make(0).fit(X, pos, floor=floor)
    c = make(1).fit(X, pos, floor=floor)
    assert np.array_equal(a.predict(X), b.predict(X)) and not np.array_equal(a.predict(X), c.predict(X))
    assert torch.equal(torch.get_rng_state(), torch_state) and np.array_equal(np.random.get_state()[1], numpy_state)
    assert not torch.are_deterministic_algorithms_enabled()  # the flag is restored after training


def test_early_stopping_keeps_the_best_epoch():
    X, pos, floor, _ = _clusters(seed=3)
    rng = np.random.default_rng(3)
    X_val, y_val = X[:40], rng.uniform(-1e3, 1e3, (40, 2)) + UJI_ORIGIN  # labels unrelated to X: no progress
    model = MLPLocalizer(hidden=(16,), epochs=200, patience=3, lr=1e-2).fit(X, pos, eval_set=[(X_val, y_val)])
    errors = model.history_["val_error"]
    assert model.n_epochs_ < 200 and model.n_epochs_ == model.best_epoch_ + 1 + 3
    assert model.best_epoch_ == int(np.argmin(errors))
    kept = np.linalg.norm(model.predict(X_val) - y_val, axis=1).mean()
    assert kept == pytest.approx(errors.min(), rel=1e-5)  # the restored weights are the best epoch's
    internal = MLPLocalizer(hidden=(8,), epochs=3, patience=5, validation_fraction=0.25).fit(X, pos)
    assert len(internal.history_["val_error"]) == 3  # a seeded 25 % hold-out when no eval_set is given
    with pytest.raises(ValueError, match="validation_fraction"):
        MLPLocalizer(epochs=1, patience=1, validation_fraction=0.0).fit(X, pos)
    assert "val_error" not in MLPLocalizer(hidden=(8,), epochs=2, patience=None).fit(X, pos).history_


def test_save_and_load_without_pickle(tmp_path):
    X, pos, floor, building = _clusters(seed=4)
    for model in (MLPLocalizer(hidden=(16, 8), epochs=3), CNN1DLocalizer(channels=(4, 8), epochs=3)):
        model.fit(X, pos, floor=floor, building=building)
        path = model.save(tmp_path / type(model).__name__, info={"split": "train"})
        torch_state = torch.get_rng_state()
        loaded = load_model(path)
        assert torch.equal(torch.get_rng_state(), torch_state)  # rebuilding the network draws no global randomness
        a, b = model.localize(X), loaded.localize(X)
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.floor, b.floor)
        assert np.array_equal(a.building, b.building) and repr(loaded) == repr(model)
        config = json.loads((path / "config.json").read_text())
        assert config["object"]["state"]["net_config_"]["__dict__"]["backbone"] == model.net_config_["backbone"]
        with np.load(path / "arrays.npz", allow_pickle=False) as npz:
            assert any(".net_weights_." in key for key in npz.files)
        assert sorted(p.name for p in path.iterdir()) == ["arrays.npz", "config.json"]


def test_input_rules_and_pipeline():
    X, pos, _, _ = _clusters(seed=5, n=40)
    with pytest.raises(ValueError, match="FillMissing"):
        MLPLocalizer(epochs=1).fit(np.where(X > 2, np.nan, X), pos)
    with pytest.raises(ValueError, match="Complex"):
        MLPLocalizer(epochs=1).fit(X.astype(np.complex64), pos)
    with pytest.raises(ValueError, match="eval_set"):
        MLPLocalizer(epochs=1).fit(X, pos, eval_set=[(X[:, :3], pos)])
    with pytest.raises(ValueError, match="eval_set"):
        MLPLocalizer(epochs=1).fit(X, pos, eval_set=(X, pos))  # a list of pairs, as in XGBoost
    with pytest.raises(ValueError, match="epochs"):
        MLPLocalizer(epochs=0).fit(X, pos)
    raw = np.where(X > 2, np.nan, X - 70)  # dBm with missing readings
    model = create_model("mlp", hidden=(8,), epochs=2, patience=1, preprocess=FillMissing(-104))
    model.fit(raw, pos, eval_set=[(raw[:10], pos[:10])])  # the pipeline fills eval_set too
    assert isinstance(model, LocalizerPipeline) and np.isfinite(model.predict(raw)).all()
    one_d = MLPLocalizer(hidden=(4,), epochs=2, patience=None).fit(X, pos[:, 0])
    assert one_d.predict(X[:5]).shape == (5,)
    with pytest.raises(ValueError, match="batch_size must be >= 2 with batch_norm"):
        MLPLocalizer(hidden=(4,), epochs=1, batch_size=1).fit(X, pos)  # not torch's opaque BatchNorm error
    MLPLocalizer(hidden=(4,), batch_norm=False, epochs=1, batch_size=1, patience=None).fit(X, pos)  # no BN: fine
    with pytest.raises(ValueError, match="widths"):
        MLPLocalizer(hidden=(4, 0), epochs=1).fit(X, pos)


def test_eval_set_accepts_a_sample_table():
    from indoorloc.core import SampleTable

    X, pos, _, _ = _clusters(seed=9, n=80)
    make = lambda: MLPLocalizer(hidden=(8,), epochs=4, patience=2)  # noqa: E731
    arrays = make().fit(X, pos, eval_set=[(X[:20], pos[:20])])
    table = make().fit(X, pos, eval_set=[(SampleTable(X[:20], pos[:20]), None)])  # y from the table
    assert np.array_equal(arrays.history_["val_error"], table.history_["val_error"])
    with pytest.raises(ValueError, match="eval_set y is None"):
        make().fit(X, pos, eval_set=[(X[:20], None)])


def test_deep_localizer_defaults_are_those_of_the_dedicated_classes():
    """DeepLocalizer("mlp") / DeepLocalizer("cnn1d") build and train exactly MLPLocalizer() / CNN1DLocalizer():
    same architecture, same seeds, bitwise-equal predictions."""
    X, pos, floor, _ = _clusters(seed=10, n=64, n_features=20)
    for generic, dedicated in ((DeepLocalizer(epochs=2, patience=None), MLPLocalizer(epochs=2, patience=None)),
                               (DeepLocalizer(backbone="cnn1d", epochs=2, patience=None),
                                CNN1DLocalizer(epochs=2, patience=None))):
        generic.fit(X, pos, floor=floor)
        dedicated.fit(X, pos, floor=floor)
        assert generic.net_config_ == dedicated.net_config_
        assert np.array_equal(generic.predict(X), dedicated.predict(X))
    assert DeepLocalizer(backbone="cnn1d")._backbone_spec()[1]["projection"] == 256
    spec = DeepLocalizer(backbone="CNN1D", hidden=(4,), dropout=0.1, backbone_options={"projection": None})
    expected = {**CNN1DLocalizer()._backbone_spec()[1], "channels": (4,), "dropout": 0.1, "projection": None}
    assert spec._backbone_spec() == ("cnn1d", expected)
    assert DeepLocalizer(backbone="resnet10t")._backbone_spec() == ("resnet10t", {})  # timm's own drop_rate
    assert DeepLocalizer(backbone="resnet10t", dropout=0.2)._backbone_spec()[1] == {"drop_rate": 0.2}


def test_cnn1d_takes_channels_and_long_inputs():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 2, 16)).astype(np.float32)  # (N, antennas, subcarriers)
    model = CNN1DLocalizer(channels=(4,), kernel_sizes=(3,), strides=(1,), projection=None, epochs=2, patience=None)
    model.fit(X, rng.normal(size=(30, 2)))
    assert model.net_.backbone.conv[0].in_channels == 2 and model.predict(X[:4]).shape == (4, 2)
    deep = DeepLocalizer(backbone="cnn1d", hidden=(4, 4), backbone_options={"kernel_sizes": (5,)}, epochs=1,
                         patience=None).fit(X.reshape(30, -1), rng.normal(size=(30, 2)))
    assert [layer.kernel_size for layer in deep.net_.backbone.conv if hasattr(layer, "kernel_size")] == [(5,), (5,)]


def test_sklearn_contract_and_registry():
    assert type(create_model("mlp")) is MLPLocalizer and type(create_model("cnn1d")) is CNN1DLocalizer
    assert type(create_model("deep", backbone="cnn1d")) is DeepLocalizer
    model = MLPLocalizer(hidden=(8,), epochs=2)
    assert "backbone" not in model.get_params() and model.get_params()["hidden"] == (8,)
    twin = clone(model.set_params(lr=0.01))
    assert twin is not model and twin.get_params() == model.get_params()
    assert repr(twin) == "MLPLocalizer(epochs=2, hidden=(8,), lr=0.01)"
    sklearn = pytest.importorskip("sklearn")
    from sklearn.base import clone as sk_clone
    from sklearn.model_selection import cross_val_score

    assert sk_clone(model).get_params() == model.get_params()
    X, pos, _, _ = _clusters(seed=7, n=60)
    assert np.all(cross_val_score(MLPLocalizer(hidden=(8,), epochs=3, patience=None), X, pos, cv=2) < 0)
    del sklearn


def test_building_blocks():
    from indoorloc.methods.deep import CNN1DBackbone, MLPBackbone, MultiTaskHead
    from indoorloc.methods.deep.backbones import image_layout
    from indoorloc.methods.deep.training import _batches

    x = torch.randn(3, 2, 5)
    assert torch.equal(MLPBackbone(10, hidden=())(x), x.flatten(1))  # no hidden layers: the identity
    cnn = CNN1DBackbone(2, channels=(4, 6, 8), kernel_sizes=(5,), strides=(2, 1), output_length=1)
    assert [c.kernel_size[0] for c in cnn.conv if hasattr(c, "kernel_size")] == [5, 5, 5]  # 0.1 padding rule
    assert [c.stride[0] for c in cnn.conv if hasattr(c, "stride")] == [2, 1, 1] and cnn(x).shape == (3, 8)
    assert CNN1DBackbone(2, channels=(4, 6), output_length=3)(x).shape == (3, 18)  # 6 channels x 3 positions
    projected = CNN1DBackbone(channels=(4,), kernel_sizes=(3,), strides=(1,), projection=7, in_features=10,
                              output_length=7)
    assert projected(x).shape == (3, 28) and projected.conv[0].in_channels == 1  # 7 learned features, 1 channel
    with pytest.raises(ValueError, match="in_features"):
        CNN1DBackbone(projection=8)
    with pytest.raises(ValueError, match="pooling"):
        CNN1DBackbone(pooling=None)
    with pytest.raises(ValueError, match="activation"):
        MLPBackbone(4, activation="swish2")
    out = MultiTaskHead(8, 2, n_floors=5)(torch.randn(3, 8))
    assert sorted(out) == ["floor", "pos"] and out["floor"].shape == (3, 5)
    assert image_layout((520,)) == (1, (23, 23), True) and image_layout((3, 4, 5)) == (3, (4, 5), False)
    assert image_layout((2, 3, 4, 5)) == (6, (4, 5), False)
    assert [len(b) for b in _batches(np.arange(9), 4)] == [4, 5]  # no 1-sample batch for BatchNorm
    assert [len(b) for b in _batches(np.arange(10), 4)] == [4, 4, 2]


def test_timm_backbone_is_optional_and_not_pretrained_by_default(tmp_path):
    pytest.importorskip("timm")
    rng = np.random.default_rng(8)
    X, y = rng.normal(size=(24, 30)).astype(np.float32), rng.normal(size=(24, 2))
    model = DeepLocalizer(backbone="resnet10t", epochs=1, batch_size=12, patience=None)
    assert model.pretrained is False
    model.fit(X, y)
    assert model.net_.backbone.in_chans == 1 and model.net_.backbone.grid == (6, 5)  # 30 values -> 6 x 5 image
    loaded = load_model(model.save(tmp_path / "timm"))
    assert np.array_equal(loaded.predict(X), model.predict(X))
    with pytest.raises(ValueError, match="unknown backbone"):
        DeepLocalizer(backbone="mpl", epochs=1).fit(X, y)


def test_without_torch_the_model_is_created_and_fit_names_the_extra():
    code = (f"import sys\nfor m in {HEAVY!r}: sys.modules[m] = None\n"
            "import numpy as np\nfrom indoorloc.methods import create_model\n"
            "from indoorloc.methods.deep import *  # the package's public names need no torch\n"
            "model = create_model('mlp', epochs=1)\nassert model.get_params()['epochs'] == 1\n"
            "try:\n    model.fit(np.zeros((4, 3)), np.zeros((4, 2)))\n"
            "except ImportError as e:\n    assert \"indoorloc[deep]\" in str(e), e\n"
            "else:\n    raise AssertionError('expected ImportError')\n"
            "try:\n    from indoorloc.methods.deep import MLPBackbone\n"
            "except ImportError as e:\n    assert \"indoorloc[deep]\" in str(e), e\n"
            "else:\n    raise AssertionError('expected ImportError')")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)
