"""core.persistence stores data objects (``to_dict``/``from_dict``) such as ``apps.maps.FloorMap``.

A particle filter or a PDR fusion holding a floor plan saves as JSON + npz (no pickle) and a
loaded copy continues bit for bit; the allow-list that guards estimator classes guards data
classes too.
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest

from indoorloc.apps import Connector, FloorMap, ParticleFilter, PDRFusion
from indoorloc.core import Estimator, clone, load_model
from indoorloc.core.persistence import save


def _assert_same_tree(a, b, where="root"):
    """Exact structural equality: same container types, same array dtype/shape/values."""
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray), f"{where}: array became {type(b).__name__}"
        assert a.dtype == b.dtype and a.shape == b.shape, f"{where}: {a.dtype}{a.shape} vs {b.dtype}{b.shape}"
        assert np.array_equal(a, b), f"{where}: values differ"
    elif isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), f"{where}: keys {list(a)} vs {list(b)}"
        for k in a:
            _assert_same_tree(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same_tree(x, y, f"{where}[{i}]")
    else:
        assert type(a) is type(b) and a == b, f"{where}: {a!r} vs {b!r}"


def _two_storey_map() -> FloorMap:
    """Two floors of a 20 m x 10 m building: a corridor wall with a door on floor 0, stairs and a lift."""
    walls = {0: [[[0, 4], [9, 4]], [[11, 4], [20, 4]], [[10, 0], [10, 4]]],
             1: [[[0, 5.5], [20, 5.5]], [[1e-3, 0.1], [0.3, 1 / 3]]]}  # awkward floats round-trip exactly
    stairs = Connector("stairs", (0, 1), xy=[(19.0, 1.0), (19.0, 2.5)], cost=12.5, name="east stairs")
    lift = Connector("elevator", (1, 0), xy=(1.0, 9.0), name="lift é")
    return FloorMap(walls, bounds=(0.0, 0.0, 20.0, 10.0), connectors=[stairs, lift])


# --------------------------------------------------------------------------- FloorMap round trip
@pytest.mark.parametrize("fmap", [
    _two_storey_map(),
    FloorMap(),                                                      # no walls, no bounds
    FloorMap(bounds=(-3, -2, 5, 4)),                                 # bounds only
    FloorMap.from_polygons([[(0, 0), (8, 0), (8, 6), (0, 6)]], floor=-1),  # a basement, derived bounds
], ids=["two-storey", "empty", "bounds-only", "basement"])
def test_floor_map_to_dict_from_dict_is_exact(fmap):
    state = fmap.to_dict()
    assert isinstance(state["walls"], np.ndarray) and state["walls"].dtype == np.float64
    assert state["walls"].shape == (len(fmap.walls), 2, 2) and state["wall_floor"].dtype == np.int64
    again = FloorMap.from_dict(state)
    _assert_same_tree(state, again.to_dict())
    assert again.floors == fmap.floors and repr(again) == repr(fmap)
    for c0, c1 in zip(fmap.connectors, again.connectors):
        assert (c0.kind, c0.floors, c0.cost, c0.wait, c0.name) == (c1.kind, c1.floors, c1.cost, c1.wait, c1.name)
        assert c0.links() == c1.links() and np.array_equal(c0.xy, c1.xy)


def test_floor_map_owns_read_only_copies_and_is_shared_by_clones():
    walls = np.array([[0.0, 0.0, 5.0, 0.0], [5.0, 0.0, 5.0, 5.0]])
    bounds = np.array([0.0, 0.0, 5.0, 5.0])
    fmap = FloorMap(walls, bounds=bounds)
    walls[:] = 99.0
    bounds[:] = 99.0  # the caller's arrays no longer reach into the map
    assert fmap.walls.max() == 5.0 and fmap.bounds.tolist() == [0.0, 0.0, 5.0, 5.0]
    assert not fmap.walls.flags.writeable and not fmap.wall_floor.flags.writeable
    assert copy.deepcopy(fmap) is fmap and copy.copy(fmap) is fmap  # immutable: nothing to copy
    pf = ParticleFilter(10, floor_map=fmap)
    assert clone(pf).floor_map is fmap
    sklearn_base = pytest.importorskip("sklearn.base")
    assert sklearn_base.clone(pf).floor_map is fmap


# --------------------------------------------------------------------------- estimators holding a map
def _corridor_walk(n=40, seed=0):
    rng = np.random.default_rng(seed)
    lengths = 0.7 + rng.normal(0, 0.03, n)
    headings = np.deg2rad(8.0) + rng.normal(0, 0.02, n)  # a biased compass: the corridor walls correct it
    return lengths, headings


def test_particle_filter_with_a_floor_map_saves_and_resumes_bit_for_bit(tmp_path):
    corridor = FloorMap({0: [[[0, 0], [40, 0]], [[0, 2], [40, 2]]]}, bounds=(-1.0, 0.0, 40.0, 2.0))
    L, H = _corridor_walk()

    def fresh():
        pf = ParticleFilter(400, step_length_std=0.05, heading_std=0.05, floor_map=corridor, floor=0,
                            recovery=(0.05, 0.5), random_state=7)
        return pf.initialize([0.5, 1.0], 0.2, heading_bias_std=0.2)

    pf = fresh()
    head = [pf.step(lk, hk) for lk, hk in zip(L[:20], H[:20])]
    pf.correct([14.5, 1.0], spread=1.0)
    path = pf.save(tmp_path / "pf", info={"plan": "corridor"})
    config = json.loads((path / "config.json").read_text())
    node = config["object"]["params"]["floor_map"]
    assert node["__object__"] == "indoorloc.apps.maps:FloorMap" and "__array__" in node["data"]["walls"]

    again = load_model(path)
    assert isinstance(again.floor_map, FloorMap) and again.floor_map is not corridor
    _assert_same_tree(corridor.to_dict(), again.floor_map.to_dict())
    assert np.array_equal(again.particles_, pf.particles_) and again.saved_info_ == {"plan": "corridor"}
    tail = [again.step(lk, hk) for lk, hk in zip(L[20:], H[20:])]

    twin = fresh()
    ref = [twin.step(lk, hk) for lk, hk in zip(L[:20], H[:20])]
    twin.correct([14.5, 1.0], spread=1.0)
    ref += [twin.step(lk, hk) for lk, hk in zip(L[20:], H[20:])]
    assert np.array_equal(np.array(head + tail), np.array(ref))
    assert 0.0 < again.particles_[:, 1].min() and again.particles_[:, 1].max() < 2.0  # the walls still act


def test_pdr_fusion_with_a_mapped_template_saves_and_resumes(tmp_path):
    fmap = _two_storey_map()
    L, H = _corridor_walk(30, seed=1)
    events = list(zip(np.arange(1.0, 31.0), L, H))

    def fresh(floor_map=fmap):
        template = ParticleFilter(300, step_length_std=0.05, heading_std=0.05, floor_map=floor_map, floor=0)
        return PDRFusion(template, heading_bias_std=0.1, random_state=3).reset(0.0, start=[1.0, 2.0], start_std=0.3)

    def run(fusion, part):
        out = [fusion.update(t, step=(lk, hk)) for t, lk, hk in (events[:15] if part == 0 else events[15:])]
        return out + [fusion.update(15.5, fix=[9.0, 2.0], spread=1.5)] if part == 0 else out

    unstarted = load_model(PDRFusion(ParticleFilter(50, floor_map=fmap)).save(tmp_path / "unstarted"))
    assert isinstance(unstarted.particle_filter.floor_map, FloorMap)  # a parameter only, nothing run yet

    fusion = fresh()
    head = run(fusion, 0)
    again = load_model(fusion.save(tmp_path / "fusion"))
    for loaded in (again.particle_filter.floor_map, again.filter_.floor_map):  # template and working copy
        _assert_same_tree(fmap.to_dict(), loaded.to_dict())
    resumed = np.array(head + run(again, 1))
    twin = fresh()
    assert np.array_equal(resumed, np.array(run(twin, 0) + run(twin, 1)))
    open_space = fresh(floor_map=None)
    unmapped = np.array(run(open_space, 0) + run(open_space, 1))
    assert np.abs(resumed - unmapped).max() > 1.0  # the loaded map still constrains the cloud


# --------------------------------------------------------------------------- the allow-list guards data classes
class _Box:
    """A user data class: opts in with _save_via_dict, has to_dict / from_dict, not an Estimator."""

    _save_via_dict = True

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)

    def to_dict(self):
        return {"lo": self.lo, "hi": self.hi}

    @classmethod
    def from_dict(cls, d):
        return cls(d["lo"], d["hi"])


class _Holder(Estimator):
    _requires_fit = False

    def __init__(self, region=None):
        self.region = region


def test_user_data_classes_need_explicit_trust(tmp_path):
    path = _Holder(_Box([0, 0], [3, 4])).save(tmp_path / "holder")
    with pytest.raises(ValueError, match="trusted_modules"):
        load_model(path)
    again = load_model(path, trusted_modules=[__name__])
    assert isinstance(again.region, _Box) and again.region.hi.tolist() == [3.0, 4.0]


def _write(model_dir, obj_node):
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"format": "indoorloc-estimator", "format_version": 1,
                                                       "info": {"__dict__": {}}, "object": obj_node}))
    np.savez(model_dir / "arrays.npz")
    return model_dir


def _holding(obj_path):
    return {"__estimator__": f"{__name__}:_Holder", "params": {"region": {"__object__": obj_path, "data": {}}},
            "state": {}}


def test_data_class_paths_are_checked_before_import(tmp_path, monkeypatch):
    (tmp_path / "object_side_effect.py").write_text(
        "import pathlib\npathlib.Path(__file__).with_suffix('.ran').touch()\n"
        "class Plain:\n    pass\n"
        "class Liar:\n    _save_via_dict = True\n    def to_dict(self):\n        return {}\n"
        "    @classmethod\n    def from_dict(cls, d):\n        return 42\n"
        "class Unflagged:\n    def to_dict(self):\n        return {}\n"
        "    @classmethod\n    def from_dict(cls, d):\n        return cls()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "object_side_effect", raising=False)
    model = _write(tmp_path / "m1", _holding("object_side_effect:Plain"))
    with pytest.raises(ValueError, match="refusing to import 'object_side_effect'"):
        load_model(model, trusted_modules=[__name__])
    assert not (tmp_path / "object_side_effect.ran").exists() and "object_side_effect" not in sys.modules
    trust = [__name__, "object_side_effect"]
    with pytest.raises(TypeError, match="to_dict"):
        load_model(model, trusted_modules=trust)  # trusted and imported, but not a data class
    with pytest.raises(TypeError, match="not the class itself"):
        load_model(_write(tmp_path / "m2", _holding("object_side_effect:Liar")), trusted_modules=trust)
    with pytest.raises(TypeError, match="_save_via_dict"):  # to_dict/from_dict alone do not make a data class
        load_model(_write(tmp_path / "m2b", _holding("object_side_effect:Unflagged")), trusted_modules=trust)
    monkeypatch.delitem(sys.modules, "object_side_effect", raising=False)
    with pytest.raises(TypeError, match="to_dict"):  # an Estimator never loads through the object path
        load_model(_write(tmp_path / "m3", _holding("indoorloc.apps.particle:ParticleFilter")),
                   trusted_modules=[__name__])
    holder = {"__estimator__": f"{__name__}:_Holder", "params": {"region": {"__mystery__": 1}}, "state": {}}
    with pytest.raises(ValueError, match="unknown node"):  # a tag this loader does not know (a newer format)
        load_model(_write(tmp_path / "m4", holder), trusted_modules=[__name__])
    with pytest.raises(ValueError, match="does not hold an estimator"):  # the top level must be an estimator
        load_model(_write(tmp_path / "m5", {"__object__": "indoorloc.apps.maps:FloorMap", "data": {}}))
    holder["params"]["region"] = {"__object__": "indoorloc.apps.maps:FloorMap"}
    with pytest.raises(ValueError, match="no 'data' dict"):
        load_model(_write(tmp_path / "m6", holder), trusted_modules=[__name__])


def test_to_dict_must_return_str_keys_and_a_failed_save_writes_nothing(tmp_path):
    class Bad(_Box):
        def to_dict(self):
            return {0: self.lo}

    with pytest.raises(TypeError, match="str keys"):
        save(_Holder(Bad([0], [1])), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(TypeError, match="unsupported type"):
        save(_Holder(object()), tmp_path / "bad")


def test_data_objects_must_opt_in(tmp_path):
    """to_dict/from_dict alone (pandas, xarray) do not round-trip dtypes: only flagged classes are data objects."""

    class Duck:  # what a pandas DataFrame or xarray Dataset looks like to duck typing
        def __init__(self, table):
            self.table = table

        def to_dict(self):
            return {"a": [1.0, 2.0]}

        @classmethod
        def from_dict(cls, d):
            return cls(d)

    with pytest.raises(TypeError, match="unsupported type Duck"):
        save(_Holder(Duck(None)), tmp_path / "duck")
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": np.array([1, 2], dtype=np.int8)}, index=["x", "y"])  # to_dict would lose int8
    with pytest.raises(TypeError, match="unsupported type DataFrame"):
        save(_Holder(frame), tmp_path / "frame")

    class HalfDone:
        _save_via_dict = True

        def to_dict(self):
            return {}

    with pytest.raises(TypeError, match="lacks to_dict"):
        save(_Holder(HalfDone()), tmp_path / "half")
    assert not any(tmp_path.iterdir())  # nothing was written
