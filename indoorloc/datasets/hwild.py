"""H-WILD: complex WiFi CSI from four APs while people walk with a handheld transmitter, UWB ground truth."""
from __future__ import annotations

import re
import urllib.parse

import numpy as np

from ..core import SampleTable, requires
from ._base import Dataset

_COMMIT = "dc8763d88a3004b0241d60d06f9d32dd0cda3abb"  # 2026-05-05, the repository's latest commit
_REPO = f"https://raw.githubusercontent.com/H-WILD/human_held_device_wifi_indoor_localization_dataset/{_COMMIT}/"
_FILE = re.compile(r"(?P<room>Con|Lab|Office|Lounge)_(?P<ap>sRE\d+)_user(?P<user>\d+)_(?P<state>w|wo)\.mat")

# name -> (building code, folder, file prefix, AP names in the order of obtain_parameters.m)
ROOMS = {
    "conference": (0, "Conference", "Con", ("sRE22", "sRE5", "sRE6", "sRE7")),
    "laboratory": (1, "Laboratory", "Lab", ("sRE22", "sRE5", "sRE6", "sRE7")),
    "office": (2, "Office", "Office", ("sRE22", "sRE5", "sRE6", "sRE7")),
    "lounge": (3, "Lounge", "Lounge", ("sRE4", "sRE5", "sRE6", "sRE7")),
}
# AP positions (m) and the "ap_toward" angles (degrees) of the authors' obtain_parameters.m, AP order as above.
AP_POSITIONS = {
    "conference": ((-1.7, 3.0), (2.0, -0.6), (4.6, 3.4), (2.0, 6.6)),
    "laboratory": ((-1.7, 3.0), (2.0, -1.1), (6.0, 3.0), (2.0, 6.3)),
    "office": ((5.7, 3.4), (1.4, 9.2), (3.2, -0.3), (-1.5, 4.8)),
    "lounge": ((-0.8, 5.6), (7.2, 6.4), (4.0, 0.0), (1.6, 9.6)),
}
AP_TOWARD = {"conference": (180, 90, 180, 90), "laboratory": (180, 90, 180, -90),
             "office": (0, -90, 90, 0), "lounge": (0, 180, 90, -90)}
# Whether the room lies behind "ap_toward" (the authors' orientation_xy.m folds angles with atan, so
# a toward angle only fixes the array axis and the sign). Measured on all UWB positions of all walks:
# the fraction behind is >= 0.998 where True and <= 0.039 where False.
_BEHIND = {"conference": (True, False, False, True), "laboratory": (True, False, False, False),
           "office": (True, False, False, False), "lounge": (False, False, False, False)}


def aoa_geometry(room: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(anchors (4, 2) m, boresights (4,) rad counter-clockwise from +x, sign (4,))`` of a room.

    The AoA of H-WILD (``estimations_aoa``, ``labels_aoa``, degrees) follows ``orientation_xy.m``:
    clockwise from ``ap_toward`` for targets in front of it, counter-clockwise from the opposite
    direction for targets behind it. The boresight used here points into the room, so the angle
    counter-clockwise from it (the convention of ``indoorloc.methods.aoa``) is
    ``sign * radians(angle)``.
    """
    toward, behind = np.radians(AP_TOWARD[room]), np.array(_BEHIND[room])
    boresight = np.angle(np.exp(1j * (toward + np.pi * behind)))
    return np.array(AP_POSITIONS[room], dtype=np.float64), boresight, np.where(behind, 1.0, -1.0)


class HWILD(Dataset):
    """H-WILD: CSI at four APs from a handheld transmitter, UWB ground truth (Zhang et al., IMWUT 2023).

    Ten volunteers walked freely (slow, fast, stopping) through four rooms holding a WiFi
    transmitter, alone for about six minutes (``_wo``) and then for about three minutes with
    other people walking around (``_w``, "with interference"). Four Intel 5300 APs (3 antennas,
    30 subcarriers each) captured every packet; a UWB system labelled the transmitter's
    position. The four AP files of one walk are packet-synchronous (same packet count, verified
    on all 43 walks), so one sample is one packet seen by all four APs. In the conference room,
    laboratory and office the four files also hold identical UWB positions; in the lounge each
    file carries its own interpolation of them (distance of a file's position from the four-file
    median: median 7 mm, 99th percentile 7.3 cm, 201 of 42,554 packets above 10 cm, at most
    1.15 m), so ``pos`` is the per-packet median of the four files' positions.

    ========= ============ ============ =============================================
    building  room         size         description (authors)
    ========= ============ ============ =============================================
    0         conference   8 m x 8 m    simple, line of sight
    1         laboratory   9 m x 10 m   simple, line of sight
    2         office       9 m x 11 m   complex, strong multipath
    3         lounge       11 m x 14 m  complex, strong multipath and NLOS
    ========= ============ ============ =============================================

    ``building`` holds the room (``meta["building_names"]``); rooms have separate frames and
    different APs, so positions and AP columns of two rooms must never be compared.

    ``features="csi"`` (default): ``X`` (N, 12, 1, 30) complex64, modality ``csi``: the receive axis
    stacks AP 1-4 x antennas 1-3 (``meta["rx_names"]``; ``meta["antenna_anchor"]`` maps each row to
    its AP 0-3, in the AP order of the authors' ``obtain_parameters.m``: sRE22 or sRE4, sRE5, sRE6,
    sRE7). Values and antenna order as stored (the 5300's scaled CSI); the order within a packet's
    90 values is antenna-major (MATLAB ``reshape(csi, 30, 3)``). With a single room,
    ``meta["anchors"]`` (4, 2) holds the AP positions. The sources state neither the bandwidth nor
    the subcarrier indices, so ``meta["subcarriers"]`` is not set. Antenna order, measured on
    the data: the phase step between adjacent antennas grows with the sine of the UWB angle
    counter-clockwise from ``ap_toward`` in 166 of the 172 AP files (fitted slope median 2.49
    rad, about 0.4 wavelength spacing); the other 6 are office sRE22 files whose phase shows no
    angle dependence (fit strength 0.11-0.14 against a median of 0.69). The antenna index thus
    runs along ``ap_toward`` + 90 degrees. For the APs whose room lies behind ``ap_toward``
    (``aoa_geometry(room)[2] > 0``), reverse the three antennas before using the CSI with the
    boresights of ``aoa_geometry`` in ``indoorloc.methods.aoa``. The 5300's per-chain phase
    offsets are not removed.

    ``features="aoa"``: ``X`` (N, 4) float64, modality ``aoa``: the authors' 2-D FFT angle estimates
    (``estimations_aoa``) converted to radians counter-clockwise from each AP's boresight, the
    convention of ``indoorloc.methods.aoa``; ``meta["anchors"]`` (4, 2) and
    ``meta["anchor_orientations"]`` (4,) give the AP positions and boresights (``aoa_geometry``).
    Needs a single room. Check of the conversion on all 119,292 packets: the authors'
    ground-truth angles (``labels_aoa``), converted the same way and triangulated, land within a
    median 0.3-4.9 cm of the UWB position in every room (90th percentile at most 19 cm). The
    estimates are much noisier (median error 6.5-12.1 degrees per AP, 15-42 degrees for sRE22,
    which the authors warn about).

    ``pos``    UWB (x, y) in metres, the room's frame (the authors quote tens of centimetres accuracy).
    ``groups`` ``user`` (volunteer: 1-5, and 1-8 in the lounge), ``interference`` (True for ``_w``
               walks), ``trajectory``
               (``"lounge/user3_w"``) and ``time`` (packet index within the walk; the files hold no
               clock).

    There is no official split; ``groups["user"]`` gives user-disjoint splits and
    ``groups["interference"]`` a clean-to-crowded shift.

    Parameters
    ----------
    environment : "all" (default), "conference", "laboratory", "office", "lounge", or a sequence.
    users : None (all) or an iterable of volunteer numbers.
    interference : None (both), True (``_w`` walks only) or False (``_wo`` walks only).
    features : "csi" (default) or "aoa".

    Only the files of the selected walks are checked and downloaded (4 AP files per walk,
    2-6 MB each; 568 MB and 119,292 packets for all 43 walks). Reading MATLAB v7.3 files needs h5py
    (``pip install 'indoorloc[datasets]'``).

    References
    ----------
    Zhang, T., Zhang, D., Wang, G., Li, Y., Hu, Y., Sun, Q., Chen, Y., "RLoc: Towards Robust Indoor
    Localization by Quantifying Uncertainty", Proceedings of the ACM on Interactive, Mobile, Wearable
    and Ubiquitous Technologies 7(4), 1-28, 2023. https://doi.org/10.1145/3631437

    Data: https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset
    """

    name = "hwild"
    urls: dict = {}   # relative path -> raw GitHub url, filled from _MANIFEST at the end of this module
    files: dict = {}  # {"all": ((relative path, sha256), ...)}, likewise
    meta = {
        "modality": "csi",
        "units": "Intel 5300 scaled CSI as stored (not calibrated)",
        "crs": "local (one frame per room; see building)",
        "pos_names": ("x", "y"),
        "pos_units": "m",
        "csi_axes": ("rx", "tx", "subcarrier"),
        "device": "Intel 5300 NICs (Linux 802.11n CSI Tool), 4 APs x 3 antennas; UWB ground truth",
        "buildings": (0, 1, 2, 3),
        "license": "not stated by the repository (cite the RLoc paper)",
        "doi": "10.1145/3631437",
        "citation": "Zhang, Zhang, Wang, Li, Hu, Sun, Chen, RLoc: Towards Robust Indoor Localization by "
                    "Quantifying Uncertainty, Proc. ACM IMWUT 7(4), 2023",
        "url": "https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset",
    }

    _max_label_spread = 0.1  # m; the median per-walk spread is at most 4.8 cm on the real files

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, environment="all",
                 users=None, interference=None, features: str = "csi"):
        super().__init__(root, download=download, verify=verify)
        wanted = [environment] if isinstance(environment, str) else list(environment)
        wanted = list(ROOMS) if "all" in wanted else [str(e).lower() for e in wanted]
        if not wanted or set(wanted) - set(ROOMS):
            raise ValueError(f"unknown environment {environment!r}; choose from {sorted(ROOMS)} or 'all'")
        if features not in ("csi", "aoa"):
            raise ValueError(f"features must be 'csi' or 'aoa', got {features!r}")
        if interference not in (None, True, False):
            raise ValueError(f"interference must be None, True or False, got {interference!r}")
        self.environment = tuple(r for r in ROOMS if r in wanted)
        if features == "aoa" and len(self.environment) != 1:
            raise ValueError("features='aoa' needs a single environment: the AP geometry differs per room")
        self.users = None if users is None else tuple(sorted({int(u) for u in users}))
        self.interference, self.features = interference, features
        self.files = {"all": tuple(e for e in type(self).files["all"] if self._selected(e[0]))}
        if not self.files["all"]:
            raise ValueError(f"no walk matches environment={environment!r}, users={users!r}, "
                             f"interference={interference!r}")

    def _selected(self, rel: str) -> bool:
        m = _FILE.fullmatch(rel.rsplit("/", 1)[1])
        room = next(r for r, (_, _, prefix, _) in ROOMS.items() if prefix == m["room"])
        return (room in self.environment and (self.users is None or int(m["user"]) in self.users)
                and (self.interference is None or (m["state"] == "w") == self.interference))

    def _parse(self, paths, split):
        h5py = requires("h5py", "datasets")
        paths = paths if isinstance(paths, list) else [paths]
        walks: dict = {}  # (room, user, state) -> {ap name: path}
        for (rel, _), path in zip(self._entries(split), paths):
            m = _FILE.fullmatch(rel.rsplit("/", 1)[1])
            room = next(r for r, (_, _, prefix, _) in ROOMS.items() if prefix == m["room"])
            walks.setdefault((room, int(m["user"]), m["state"]), {})[m["ap"]] = path
        keys = ("features_csi" if self.features == "csi" else "estimations_aoa", "uwb_coordinate_x",
                "uwb_coordinate_y")
        order = lambda walk: (list(ROOMS).index(walk[0]), *walk[1:])  # noqa: E731  rooms in ROOMS order
        parts = []
        for (room, user, state), by_ap in sorted(walks.items(), key=lambda kv: order(kv[0])):
            aps = ROOMS[room][3]
            if set(by_ap) != set(aps):
                raise ValueError(f"{room} user{user}_{state}: AP files {sorted(by_ap)}, expected {sorted(aps)}")
            per_ap = []
            for ap in aps:
                with h5py.File(by_ap[ap], "r") as f:
                    per_ap.append({k: f[k][()] for k in keys})
            parts.append(self._walk(room, user, state, per_ap))
        cat = lambda key: np.concatenate([p[key] for p in parts])  # noqa: E731
        meta = {"buildings": tuple(ROOMS[r][0] for r in self.environment),
                "building_names": {ROOMS[r][0]: r for r in self.environment},
                "ap_names": {ROOMS[r][0]: ROOMS[r][3] for r in self.environment}}
        if len(self.environment) == 1:
            anchors, boresight, _ = aoa_geometry(self.environment[0])
            meta["anchors"] = anchors
        if self.features == "aoa":
            meta.update(modality="aoa", units="rad", anchor_orientations=boresight,
                        feature_names=tuple(f"AP{k + 1}" for k in range(4)))
        else:
            meta.update(rx_names=tuple(f"AP{k + 1}/ant{a + 1}" for k in range(4) for a in range(3)),
                        antenna_anchor=np.repeat(np.arange(4), 3))
        return SampleTable(cat("X"), cat("pos"), building=cat("building"), ids=cat("ids"),
                           groups={key: cat(key) for key in ("user", "interference", "trajectory", "time")},
                           meta=meta)

    def _walk(self, room: str, user: int, state: str, per_ap: list[dict]) -> dict:
        name = f"{room}/user{user}_{state}"
        n = per_ap[0]["uwb_coordinate_x"].size
        if any(ap["uwb_coordinate_x"].size != n for ap in per_ap):
            raise ValueError(f"{name}: the AP files disagree on the number of packets")
        uwb = np.stack([np.column_stack([ap["uwb_coordinate_x"].ravel(), ap["uwb_coordinate_y"].ravel()])
                        for ap in per_ap])  # (4, n, 2)
        pos = np.median(uwb, axis=0)  # exact where the files agree; robust to one stray lounge label
        if np.median(np.linalg.norm(uwb - pos, axis=2).max(axis=0)) > self._max_label_spread:
            raise ValueError(f"{name}: the AP files disagree on the UWB positions (not the same walk?)")
        if self.features == "aoa":
            _, _, sign = aoa_geometry(room)
            X = np.column_stack([ap["estimations_aoa"].ravel() for ap in per_ap]) * sign * (np.pi / 180)
        else:
            # h5py sees MATLAB's (n, 90) complex matrix as a (90, n) compound array of (real, imag)
            csi = [(ap["features_csi"]["real"] + 1j * ap["features_csi"]["imag"]).T for ap in per_ap]
            X = np.stack(csi, axis=1).reshape(n, 4 * 3, 1, 30).astype(np.complex64)
        return {"X": X, "pos": pos, "building": np.full(n, ROOMS[room][0], dtype=np.int64),
                "ids": np.array([f"{room}-user{user}_{state}-{t:05d}" for t in range(n)]),
                "user": np.full(n, user, dtype=np.int64), "interference": np.full(n, state == "w"),
                "trajectory": np.full(n, name), "time": np.arange(n, dtype=np.int64)}


def _layout(manifest: str) -> tuple[dict, dict]:
    """``files`` and ``urls`` from the manifest below: a folder line, then ``<file> <sha256>`` lines."""
    entries, folder = [], ""
    for line in manifest.strip().splitlines():
        if line.endswith("/"):
            folder = line
        else:
            filename, sha = line.split()
            entries.append((folder + filename, sha))
    return {"all": tuple(entries)}, {rel: _REPO + urllib.parse.quote(rel) for rel, _ in entries}


# sha256 of every walk file at _COMMIT, computed while streaming it from GitHub and checked against
# GitHub's git blob hash.
_MANIFEST = """
Conference/
Con_sRE22_user1_w.mat 19e854ca9ba4c8adb52d2e0f0b5b40172cb5dd4aca26638fe73e6fefd49a587b
Con_sRE22_user1_wo.mat a6d79dc3eefc8dc24b96271e02c83950112b8b2065bfa3398791b700aa8421a5
Con_sRE22_user2_w.mat 8d7b8c2312ec7da9b1c95c2786149ed5fa9c8ac0cf48e55741fa7e16f54543e1
Con_sRE22_user2_wo.mat 0f6441fc63a76354eb9ae46e0ac0a6a2b6152822155ee6e228d7158b8a431ac3
Con_sRE22_user3_w.mat c051cb9289429ab05204e0bac5d5172359f56eba298c01b3211fd5f8f493db5c
Con_sRE22_user3_wo.mat f0cfd38244d33c1e8ee04a00110eea2e8522e685d0416d35b1e502fbf5ab0db4
Con_sRE22_user4_w.mat 238f8b04bbc2c7cc38d76ba9ab4d02d9ac94e560e62e3ac080fc461366634d25
Con_sRE22_user4_wo.mat d2e139f39edefe6a9ace43f7c121bea842a2f18e9c268869607fe19e8ded7e24
Con_sRE22_user5_wo.mat f4ffd160aaefd7e03a6486820bd940a7a3e58488f8b7c1d8574252e49087c999
Con_sRE5_user1_w.mat 156d1f277fc7bc5a6bd6203fe8701f09c0a352fe41049ce7df9ef037206923a4
Con_sRE5_user1_wo.mat d4bbff04e1c3f7f9e4aa928967176d32dfc1c469e5039ca22a48ac4e4eb0159a
Con_sRE5_user2_w.mat 6e01c31bf93e52177f7deeabe3a2162cdd4527a8da7e8ea8bdc4b5ff5bbaf39a
Con_sRE5_user2_wo.mat cb98d16c4851c84629dc79198c6f588dcd1316f2ec6cd6bbcff750d0d69a9a4e
Con_sRE5_user3_w.mat e5168184dd523ff2b6062856bcdd619aa5436faa7971e029ac724f84a5bac002
Con_sRE5_user3_wo.mat 51d4934751f04e3c5742062810145ed87a012ea32847630b742b800f02e5c8c9
Con_sRE5_user4_w.mat 14f27773a68561569197ac7c07b52d4d297845012afb6e993955c1ef1a1dc33a
Con_sRE5_user4_wo.mat 6a139bd6e4c65c1eff591a1ae0c83f1bd355eb0254daabea845bde0d1036c34f
Con_sRE5_user5_wo.mat 2946994a5077f06e704bfdb335a680f552743ce54e7d287001e922ad9b31ff6e
Con_sRE6_user1_w.mat b89c312ded72acee65034dc65730eade935e5861d7368a0bfea3fc491d5c1292
Con_sRE6_user1_wo.mat b007654b60329f2fab836df2c908db5ced33bbbbc2e4d0f1b3ef5fa0dd1e4073
Con_sRE6_user2_w.mat 9fcf65aad6bcce517efb682a79c63b44f2777fc6923b104c13133fbd4bd3317a
Con_sRE6_user2_wo.mat 2ddc1b60fe8a193c64df453260e83e99513f6c77c239038610c40967f9c8c79e
Con_sRE6_user3_w.mat 3483bf74ec2e52e839e9c39acae061c2fc0b82ca4d3971ea40087c39ce9deb62
Con_sRE6_user3_wo.mat 0c6198b2e369c3f61d31866fb77785716a8e60a0d89273cef8506fb374b119ec
Con_sRE6_user4_w.mat 53e5049ed0aee8affd3030268ebd02c734dc07b5dcbe74a5f09d7be6de83acfc
Con_sRE6_user4_wo.mat 9cf4c95e6ade53727a4632eba4b1c9316d6f80d21f085acb17e6aaa1bd5c2813
Con_sRE6_user5_wo.mat 80e3b9b876ca0966026f30a26291544de3dbdf2f5400a8783c64c2bec0c734da
Con_sRE7_user1_w.mat d8d2a271506ca388af609e96dce98b3289e9ace2077408881dd05b29fa0d6f40
Con_sRE7_user1_wo.mat f55636569cf953dbd257f53650c21ba24c8de20c4e87f5ca309cb45716b7f9b5
Con_sRE7_user2_w.mat 6bfff4defbffd7252660efdffd4db066bfa423843236247be64167d77821e64e
Con_sRE7_user2_wo.mat fcb7f12d77eedab114617b546fce303435c65d9c5d48a4117a36036a1ebe5303
Con_sRE7_user3_w.mat 5bcbe3ec7cb32471fc77def5f99430eabfeedcc8b4d4cfaf2ce9bdb2e88ff875
Con_sRE7_user3_wo.mat 4ba8a0fb8c2d6150833fc087c695dc4aaf0954b1806c2be9f5c96e0a66cff71a
Con_sRE7_user4_w.mat 75c721a580ad49f1d032f776733aad316113369d4242a8f656018f2c02dacba6
Con_sRE7_user4_wo.mat e824a0d4a29c83bf12e8963deb6a523d582a20115c1ea2f0dce3284bc566f007
Con_sRE7_user5_wo.mat aa7e6e81a7805b6c2ab5db59211a8cd6c27df2dc3e71790591f69df3ffc64179
Laboratory/
Lab_sRE22_user1_w.mat 8efc69cc2852159403ebc8b8c8785d154c614f494601cf326e36a84d150c4c5c
Lab_sRE22_user1_wo.mat 16e0f16c8d4c786d8560e8c3baa9af60d5656c50b5e2e71ee5c8d5a3d949e912
Lab_sRE22_user2_w.mat 4cedf7fb79d4ad6bfd912d5d5e07ae5aa18996c1f2cf19b7ae195d3aab19747f
Lab_sRE22_user2_wo.mat e4d29c052394837b098e9e0beef068c063cff1aebf5d0d433e6791291ffc63ac
Lab_sRE22_user3_w.mat faa893f3ea60e85295bc475360ca24e8895449e8b706a6e0d8d383eb8a5f452a
Lab_sRE22_user3_wo.mat d5a0486ee7673030ae1f30357c91e557c51091e263a5a038a58dc55d425301f1
Lab_sRE22_user4_w.mat f8468feda03c1c7f28c42d4b571ffe5ee8919fe5a8e5949233b29bc5fe072acb
Lab_sRE22_user4_wo.mat 82c2ee2381b361c88cd058194b79b42e7544ff70ecce398fff2a1d893c31962c
Lab_sRE22_user5_w.mat bddd1adc34da04e8eac93c5be8bc1f87f3324a48586fd5deda3ab10e23b6870c
Lab_sRE22_user5_wo.mat 90cd4bebf70276f4851a60fe1479ee0035c5ba0b504081512409804e4a5d9ddc
Lab_sRE5_user1_w.mat a9e0e98b9ebed02d482fd8417c7b92efdb33757586db29e2d742eec490661f56
Lab_sRE5_user1_wo.mat 0b10aeb17611dfa71a4b4ae2717b0c681d26b326570184a56cddce941eb0c515
Lab_sRE5_user2_w.mat 4a38d602ccc16252976c82f1ede28df31acf37b309e21db4c3672478a6f65f8f
Lab_sRE5_user2_wo.mat 633a6a337078e2a9e03a7ca20fc6e8368f631c7996f653f986419bb7fcbf8a37
Lab_sRE5_user3_w.mat c3e05bb8371a5d8dea459ea66e376551a9f0d71b79207b90e1c078e00baf0407
Lab_sRE5_user3_wo.mat f4687da0aa2325aa1fd796232669099a9722a5ed536e12841eea94cdef3d05ff
Lab_sRE5_user4_w.mat 6b2b65b35b34142610245657d19a6b0238571f06866f7ef11b13fbb56d5ca613
Lab_sRE5_user4_wo.mat bdd08302fcab8bdad8c346e5a040b9b4daf2c79fb53430c74f0cf693df7ce615
Lab_sRE5_user5_w.mat 7a34bc7606dacb8d1e0d26b008727a6cd9df8f60c0097fc86ce0cc5936b9919b
Lab_sRE5_user5_wo.mat fdac9c96e287ef50b34de74b19969c78fef26a6def102fd23e8923c59d51dae6
Lab_sRE6_user1_w.mat 9ad86ba88d45dfaa61efffe3c1544d817d934a51fe7007675075cd1788a37633
Lab_sRE6_user1_wo.mat 52ede0b88c58b2fe2bf877a8a4bfb63cb0a0ab32e97550f4e0bc43d8d686cde4
Lab_sRE6_user2_w.mat 665d8b283dead45fa61ff4247196e18711ad8424089a4dfae131b9ff75a0f464
Lab_sRE6_user2_wo.mat 4dfe3e88cc675af06826d1c5d2c03e1de6df6ae2a45735beb9e928b8684bf2f9
Lab_sRE6_user3_w.mat 70a7f288b469bd468cd933e4a6a3d41de4c542dcbe88a66f828da5b5af3c17e3
Lab_sRE6_user3_wo.mat c83e5c61bbf843537d1b8c63549ced2a6c02e879564e028e9955de7ed6dceb7d
Lab_sRE6_user4_w.mat 2fd516b85b195b0b2889c1ebd8b26c1169468a3194da86c2dc44f6ebb1c85af8
Lab_sRE6_user4_wo.mat 1d7c030f2727d1e6955fd8a062ffe63f0b43fc89b2497e7b34db1a426ff2c0d9
Lab_sRE6_user5_w.mat b265120e6f807c8b10823029fab4aae4757514fe167a94961cdfe1f85c39967a
Lab_sRE6_user5_wo.mat a21126408c86f1902db25e9cf816251446c0e8cbe97c8b58032ec242e7f2575f
Lab_sRE7_user1_w.mat 10ad490b95725c37813e14e5b9d8b86252c84171093362cf519ae37a0f45d7a0
Lab_sRE7_user1_wo.mat 5ff3a55ac3a3d4902c9e258dc0136e4778777b7eb5aefe87c9136641fb57c3fc
Lab_sRE7_user2_w.mat 9d5df3be187664cb82f4584a99fee6a2180670c44d38227f8e2013c7d6a51d3c
Lab_sRE7_user2_wo.mat e69c48ad53181724421383ffb0a6546777007003959538d9e23201ec3e616578
Lab_sRE7_user3_w.mat 28bfb2c90856848f142ff912a1a722dbcec9ea96d5141c64979705d4c25879e9
Lab_sRE7_user3_wo.mat 9e8b8f8d30b44d3092b3991563ff53669d836cc9f3fe106ff34cf43c684dd7de
Lab_sRE7_user4_w.mat 63231340dd1eabd74c1ce6a79939aaee90000d95f924d2789bdba5d69d631149
Lab_sRE7_user4_wo.mat 6acb77b04190459f5d134f3894435b0dbc54f004395139cf1430a26e84b4653e
Lab_sRE7_user5_w.mat 9443d7dd8020380f8b2db77c08cb749142cecc0976be3a862d2d91abb47a62ed
Lab_sRE7_user5_wo.mat 3229405a4148f5dcc3478c566202c73b228e10b262da42fb72807614fea3f2d8
Lounge/
Lounge_sRE4_user1_w.mat 5500e414e9d70bb9e1423567c9d4748928646681a92fedfe4fdc51d62eb54470
Lounge_sRE4_user1_wo.mat f917bc69c1a351980e41377814db3d0e54b9dcb5c6f3b919f948060747d28a49
Lounge_sRE4_user2_w.mat ce2544313beedf522f8d5cdef6cf1d49ed126029c4038d0ffee8b10cc53e68be
Lounge_sRE4_user2_wo.mat 645b03a4e6d81af0494c906c94edc9fa78c63f013f721750470ddbda7d4a340f
Lounge_sRE4_user3_w.mat 928d606832c92bc7d0d875242f34ad202355ec5126ba366ab4d4a36c1c6740fc
Lounge_sRE4_user3_wo.mat 81b33d0278673510800122eb6d27273568756b7bf4d5f3b334b27680c7757f88
Lounge_sRE4_user4_w.mat a65fb89e8abd5241df17186d6a722b2614fd5b8ce243ded335255282a2b4f7ee
Lounge_sRE4_user4_wo.mat d8e9fe18a8648ef623c32015fe6f9e8ccf8215a9dc2632e668f7039ee13e6bdc
Lounge_sRE4_user5_w.mat 19c73f0ef66a804fe853d160421b7a7459269ad9ef3a8063e4c85faba182f1c7
Lounge_sRE4_user5_wo.mat 15e140689da3dc12ec00098e807960c8539d44f7fbc45d97268dfe1035608a76
Lounge_sRE4_user6_w.mat cffd8e7c606682c0fb920efdc9c418473dbe2cf76c0a3cd17b7ac5747f702417
Lounge_sRE4_user7_w.mat 504f7de20abe0d3022c53df7bef436a5e3b27282d183a1aff364920eecb1be1c
Lounge_sRE4_user7_wo.mat 21722c215f264881483b5e6cc4f623f14f37b44f55630b8340c482f8ad4ced2f
Lounge_sRE4_user8_w.mat fe9f6439c3b7e2f0892f1c06537897d347227de73a0d3f4f8e5f57942871629d
Lounge_sRE5_user1_w.mat 86b9708078ce02ca19d574600fb7750c32f61d9446d6f20c5d4f57d20eee8bc8
Lounge_sRE5_user1_wo.mat d84841045baf9b6130f46c8f51f505e3b873f0c4649a21037fa1ea4cdb262e62
Lounge_sRE5_user2_w.mat cedabb00b9fead2142f22ea86a2b05f866943c624de46adf5e3de6f146816cfc
Lounge_sRE5_user2_wo.mat 5c23a61a0177cb9b22b6f802fc6b293cfb80bfec3ef0798a1b929bd8ddfd4d4f
Lounge_sRE5_user3_w.mat cfbbcfdc029bba19da4e27b28a1b71a1ab7db2ea7f6c06e76c859b6ba16a8085
Lounge_sRE5_user3_wo.mat 58ec13ee994f890fb6e4740397c055efcbc8614d17363a973d032402c03ffa07
Lounge_sRE5_user4_w.mat 0fc7a6604c0e00d873bf53ebd9e09d23c1f3cf5f9b8770a3c075f723861ec1f0
Lounge_sRE5_user4_wo.mat ee1f91741c10dceb441b867b39d6390e0e70ccc27b33b267270dc4bb824c42bb
Lounge_sRE5_user5_w.mat 09ba1a5f639bdc7daf4adb8b1368c7ecf81c1c3d672da982db5ed1c4a6c2262b
Lounge_sRE5_user5_wo.mat 85858b0d355c8ea628f738906db57a23e34fbe227e6ec110991f23e6471f1d51
Lounge_sRE5_user6_w.mat 4eadb7af10e688330a60ce041b0ab59c5770e1d2cfb53bfc9694149dce565303
Lounge_sRE5_user7_w.mat 3999bd9f28e64576d6a806ee4614f2385695a9efd7572c6ad281e5e12daa27b5
Lounge_sRE5_user7_wo.mat 515d7ae20610303e6d0d42339ea579e7e8e6048675f04d9d55af73ef4a91ff16
Lounge_sRE5_user8_w.mat f38314e5d99e6a74434488ae70523f26afafb897b0d1085e22dd35ff396af386
Lounge_sRE6_user1_w.mat 73a6615e49829bafec5db180b68e7de856c4eea8f9826a60f00e525f7bcda634
Lounge_sRE6_user1_wo.mat 6697f7e6ec06014afef5ae93aebe45c8598f0163d4dfc63512512d8ae0a49b7b
Lounge_sRE6_user2_w.mat 5cac5df7609d1f032cded66e57c2dbe626257a4e203e82e2f72748460472d500
Lounge_sRE6_user2_wo.mat 750a36c5cf98fef0751c98ed627815bb7429823f3e51fdebd9e1f178b1c43038
Lounge_sRE6_user3_w.mat f3480f5017f0edbd510152c7b6e3995efde6ae7d89611a6724e8530af36e8f0c
Lounge_sRE6_user3_wo.mat 9a72d0dcfe643f5397fd826a6da455b55a7ed097094c719c237ed11db946a90d
Lounge_sRE6_user4_w.mat c5497175f9f2ea2be1b5bf58baca39a410db710004838d519c8f5053b8302f8a
Lounge_sRE6_user4_wo.mat 244588e50a612d331660348a9e223ced4ae33b91d0b3ca2d92d9f882224affe2
Lounge_sRE6_user5_w.mat 90eaa68367c27469a8dcac102d2c148e76acb4129566e71f940d65ea49625c03
Lounge_sRE6_user5_wo.mat e2e5582113bb9afb3040a13e0eacd225395848bb19c1d55221956c9e40c8457f
Lounge_sRE6_user6_w.mat 8e1b37469b84c5dff4dfa4cd0682c2a8cde3c08c869a32461b7c532a315c3e9c
Lounge_sRE6_user7_w.mat d080cbe56cefa3ec6d051ca9a2db20a4b50d78dfc1bb58deaf4c8e528739b9ad
Lounge_sRE6_user7_wo.mat 08b74e52147dc0e916828edd67a5779dd159d17c3b56afe3012dddbeb6a0da06
Lounge_sRE6_user8_w.mat 9e7a742ad6ae7ba5aa94d7e858c2d968ecade8ad9162c3f39dd7f50efe5429a5
Lounge_sRE7_user1_w.mat a2e82d9e1ba9fd032cfb9cab4707aa456aeb65ffe19d4dfaeeeec9d4c8fd3ca4
Lounge_sRE7_user1_wo.mat fb5c330128769aa948723d33c8d8c9bb574e0312452f583b3898c963d00cf527
Lounge_sRE7_user2_w.mat fae68590d87fc1c2809581b1442707bd8a5c8db618cd145435db93a099bbf766
Lounge_sRE7_user2_wo.mat ab3b1e30aafd09b14cc258c9c143995222d9352181396127a9c46f402c9899db
Lounge_sRE7_user3_w.mat 49742ba0a56e376139511bf7659acb73168415c2f9185213d268dc585d63d7aa
Lounge_sRE7_user3_wo.mat 89dfd0ceab3e6fda4e635b1a4a1ac1ff63c3811e4abf34f61fcddf9ff17f262c
Lounge_sRE7_user4_w.mat 1c792698f717e297b837eb4f7ed4c2c4db3c65e192ff8bf1183c40cf580d6b65
Lounge_sRE7_user4_wo.mat e6bdd00f6b8445a8a1f37092a35f2d55d4700473c6fdaf76630d9f4d700eee52
Lounge_sRE7_user5_w.mat c37d1dd5b365c4c8f0398b75a7f610dcd473ce1c8b1113f7b21d76050ed2727c
Lounge_sRE7_user5_wo.mat de2ba11c98c49d98a63caf0954956545c786c4f6c8ec3d0e773f433b0df68a9c
Lounge_sRE7_user6_w.mat 95db17112c258d0b80e4bc97a1a197527e85475bd98c0d7c0341c35ef0a1bc21
Lounge_sRE7_user7_w.mat bf731f74023a5c41139e2080d98583dbc82e766ced68f2b5a704b28064577206
Lounge_sRE7_user7_wo.mat 5e33dd21011e4599a7112af5d6922ae718726f891daf2003b7c9220816665302
Lounge_sRE7_user8_w.mat df3a4acec17f07d64b5b1c89805a423c6c8457693aa84a7143a22db9d7d407e8
Office/
Office_sRE22_user1_w.mat 31b4bb6f1f59e103c9a3d74d8c2b9d60e93b60c735cfd3f51e6e175856cb577d
Office_sRE22_user1_wo.mat b4c4df5216f02616bf5a1c56905d2e790cf94a3fdfbbdd96951ef36c311f2692
Office_sRE22_user2_w.mat 0be1c4d61c94ffc49beda947ad748cc145ecf7b9ab3ecc94977d92462ac0affd
Office_sRE22_user2_wo.mat e005ff4dd53e570ca3ebe05fb759b23f0b18e067e904e2b810410fe1de01bfcf
Office_sRE22_user3_w.mat 318a15ec4b6237fd3b3d09fce81f097df26375b966f876159c7a6090c667df7c
Office_sRE22_user3_wo.mat 3987a93e27b3f3a6011388ca929695d03ede09c2db1850d47e108625769192ac
Office_sRE22_user4_w.mat cbf4cfe22d21ac1721ce45cba25d94c2613f02a2b53776ef6eb81099cfd08588
Office_sRE22_user4_wo.mat 1a9c0640f7da7079c1b6d164f52f707a7aa0061eaae7f07ee1092f37d15796eb
Office_sRE22_user5_w.mat f758742aa54cf3bc9a20fe1abbe8e788a77dc8ae74f3c0134bab01133c5fe061
Office_sRE22_user5_wo.mat ab25af70481d50c095e40e129a164c896dc89d91f13fa7a838a0433ec7412414
Office_sRE5_user1_w.mat 4cd71dee40ec667d86c97efc0f2bea8ab56a336c150fa5a9a5adbe801ece0107
Office_sRE5_user1_wo.mat b3dbb4d3f450bc7793dcccc0bfd894be4120e20e0c8f747c0d6bf6c6208d0cb0
Office_sRE5_user2_w.mat 6e6d661b666b59bd1f08e576b1a9439f309874fc013675b92eb2b9dd55da6d82
Office_sRE5_user2_wo.mat 529da6d37dfaf96fd74e61cf7ceb495b784080ffa67d35d5c2fcdaf42f9aa1e6
Office_sRE5_user3_w.mat 94710a1287ec9a266665948017dad6c4f2f5101a4256de1c46344e7c029a34b1
Office_sRE5_user3_wo.mat 51be62f3962956fa345f637767c0f77be7ecfe0d1ea5a25b3e9cfd66de1b92ef
Office_sRE5_user4_w.mat e593e23a1f3d0eb37617195de2b92acdc2ddcb7caae98ab132acb78b83585bfd
Office_sRE5_user4_wo.mat 966882817cf1cad18b6ea8f0f79e8fa7d56e76380be910c51100372c452f9dfb
Office_sRE5_user5_w.mat e188278ec0049499b8cb15548016127e5b64e4792ccf64dd1144ab5d2ca53412
Office_sRE5_user5_wo.mat 1d908eec03d7db0cdb7b7200ff57bf6fe61e17cfc248a9b153be27a1a90f03bc
Office_sRE6_user1_w.mat 9b864f3df7eaaf4e58651d1d5afcf9841b634364474ad66bdd86f2cd6a9cdaff
Office_sRE6_user1_wo.mat 517019ab860e9cbef6f9b40cac94d3b04c1cee93fe2a12fcf7a7832b646169a9
Office_sRE6_user2_w.mat 44b30db712559279df3ec3983da4709ae69671463c0c16170dc87196fee9a791
Office_sRE6_user2_wo.mat 40f7d52944592242e2a738014cade2aaae045a42a13339a4ccb857be07f9c324
Office_sRE6_user3_w.mat f051f7d83e446020fea60372004eb37acbe6602f1de11724468ba2d8063ce823
Office_sRE6_user3_wo.mat a3842cc6d136933dee975649cf1e2a90f4d35ee913ce94382d11ef483a65822d
Office_sRE6_user4_w.mat 9a3be1063f2c0303256fc77775ecaffc22fd4ceca9427e62b8a7878d6e4dcd59
Office_sRE6_user4_wo.mat 7230fa616964fde0156a7e26192e2043483ec0aa63726c377dbb4ad3ad38b86c
Office_sRE6_user5_w.mat 890c57d87423ba4316917ba8a9c893b48f679648086694ba4d39d929cd521612
Office_sRE6_user5_wo.mat 918253c121953cdddccd10a4c942f7490131c7d1e6f6c074dc35ca547169a4bd
Office_sRE7_user1_w.mat 3bc52d0c4205a33c2caf3e361cf8c09a4d18e3268f7409afda9969e156502b8c
Office_sRE7_user1_wo.mat 179f5216ffd63e87b4d11f5df3b477783d2f67a6c99035966bd9f7c6898c8ee5
Office_sRE7_user2_w.mat b48f361970af2a2245d780bfaa0b33aa0bfce45732b77982fdce8891efaf6b7c
Office_sRE7_user2_wo.mat 9c42db9d645ab7ffd2fb124d346e5f6a5b7b5da36933e5a5db37aa637e0fd893
Office_sRE7_user3_w.mat 5595301c963a34044e6ff8702ac7e468c9fa20cf83b6bd011642ce82df7200fa
Office_sRE7_user3_wo.mat a4870cabe4e67e4297cc33a1fb03df16bd39579e76a4c1334d2a45f30b9cadb9
Office_sRE7_user4_w.mat 39f2aea0e6aa0ddfbda495c7608b0ca5e34acf24d4d19e61b0509b49678854dd
Office_sRE7_user4_wo.mat cddc3e3c0f972662b46c0e9f117e79be65c9e974f2335dfe62e5ded5e7e17e01
Office_sRE7_user5_w.mat e51e8b19b8b9ca7b89496f17afff454965cbf8337cce6582f0cc2de284fff18b
Office_sRE7_user5_wo.mat 0ec7a887993a581f2e29daf36539734551de0f2a1894a1f3b18de23150e12a7d
"""
HWILD.files, HWILD.urls = _layout(_MANIFEST)
