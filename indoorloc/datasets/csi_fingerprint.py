"""CSI-dataset-for-indoor-localization: Intel 5300 CSI amplitude fingerprints of four rooms (Zhu et al.)."""
from __future__ import annotations

import re
import urllib.parse

import numpy as np

from ..core import SampleTable, requires
from ._base import Dataset

_COMMIT = "ec837795b64c31b95f5f338b8916f4ff3ad0dcc2"  # 2023-11-24, the repository's latest commit
_REPO = f"https://raw.githubusercontent.com/qiang5love1314/CSI-dataset-for-indoor-localization/{_COMMIT}/"


def grid_position(filename: str) -> tuple[int, int]:
    """``"coordinate715.mat"`` or ``"1006.mat"`` -> ``(7, 15)`` / ``(10, 6)``: the last two digits are the second index.

    The repository's README defines it: "the file 'coordinate 715' means position coordinate is [7, 15]".
    """
    match = re.fullmatch(r"(?:coordinate)?(\d{3,4})\.mat", filename)
    if match is None:
        raise ValueError(f"not a reference-point file name: {filename!r}")
    digits = match.group(1)
    return int(digits[:-2]), int(digits[-2:])


class CSIFingerprint(Dataset):
    """Intel 5300 CSI amplitude fingerprints at reference points of four rooms (Zhu et al., IEEE TMC 2023, TNSE 2022).

    A laptop with an Intel 5300 NIC (Linux 802.11n CSI Tool) recorded the CSI of a fixed
    transmitter at every reference point of four rooms. One file per reference point holds
    ``myData`` of shape (3 antennas, 30 subcarriers, packets); every packet is one sample.

    ========= ==================== ========== ================= ==========================
    building  area                 points     packets per point  authors' description
    ========= ==================== ========== ================= ==========================
    0         lab                  317        1,500              13.5 m x 11 m lab, NLOS
    1         meeting              176        1,500              7 m x 10 m meeting room, LOS
    2         conference           160        50                 conference room (2022)
    3         minilab              35         50                 small lab (2022)
    ========= ==================== ========== ================= ==========================

    ``building`` holds the area (``meta["building_names"]``); areas do not share a frame.

    ``X``      (N, 3, 1, 30) float32, modality ``csi_amp``: **CSI amplitude in dB,
               20 log10 |h|, exactly as stored.** Checked on meeting-room point 101 against the
               repository's ``imaginary_part`` file: with |h| from these values and Im(h) from that
               file, sqrt(|h|^2 - Im(h)^2) * sqrt(3) is within 0.01 of an integer for all 135,000
               entries (median distance 1.6e-4), and so is Im(h) * sqrt(3): both parts are the
               Intel 5300's integer readings scaled by 1/sqrt(3). Read as linear amplitudes the values
               fail this test (1 % of entries within 0.01). The sign of Re(h) is lost, so the
               imaginary-part files (lab and meeting only) cannot restore the complex CSI and are not
               loaded. The two 2022 areas
               (-14 to 38) hold values like the lab and meeting room (-9.5 to 39.2) and are assumed
               to use the same scale. A zero amplitude is stored as -inf (1,845 entries in 1,676
               packets of 34 lab files, none elsewhere); it becomes NaN, as
               ``indoorloc.signals.csi.amplitude(db=True)`` does.
    ``pos``    (i, j): the two numbers of the file name (``grid_position``), **in grid steps, not
               metres**. The repository gives no spacing and its drawings are not to scale (they
               suggest roughly 0.5-0.6 m). The first index runs across rows in the lab, meeting and
               miniLab drawings but across columns in the conference room.
    ``groups`` ``point`` (``"lab/715"``: one reference point, for splits by position) and ``packet``
               (index of the packet within its point's file, in recording order).

    There is no official split. Packets of one point were recorded together and are near
    duplicates, so a random packet split measures point recognition, not localization; split by
    ``groups["point"]`` to evaluate positions that were not in the training set. Measured with
    WKNN (k=5) on the first 50 packets per point: a random 80/20 packet split gives a mean error of
    0.004 (lab) and 0.000 (meeting room) grid steps, while holding out 20 % of the points gives
    10.93 and 5.65, worse than always answering the training centroid (8.47 and 5.16).

    Parameters
    ----------
    area : "all" (default), "lab", "meeting", "conference", "minilab", or a sequence of these.
    packets : None (every packet) or an int k: the first k packets of every point (the lab and
        meeting room hold 1,500 per point, 750k samples and 270 MB of float32 for all areas).

    Reading ``.mat`` files needs scipy (``pip install 'indoorloc[datasets]'``).

    References
    ----------
    Zhu, X., Qiu, T., Qu, W., Zhou, X., Atiquzzaman, M., Wu, D., "BLS-Location: A Wireless Fingerprint
    Localization Algorithm Based on Broad Learning", IEEE Transactions on Mobile Computing 22(1),
    115-128, 2023. https://doi.org/10.1109/TMC.2021.3073005

    Zhu, X., Qu, W., Zhou, X., Zhao, L., Ning, Z., Qiu, T., "Intelligent Fingerprint-Based Localization
    Scheme Using CSI Images for Internet of Things", IEEE Transactions on Network Science and
    Engineering 9(4), 2378-2391, 2022. https://doi.org/10.1109/TNSE.2022.3163358

    Data: https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization (MIT license).
    """

    name = "csi_fingerprint"
    urls: dict = {}   # relative path -> raw GitHub url, filled from _MANIFEST at the end of this module
    files: dict = {}  # {"all": ((relative path, sha256), ...)}, likewise
    meta = {
        "modality": "csi_amp",
        "units": "dB",
        "amplitude": "20 log10 |h|",
        "crs": "local grid (one per area; see building)",
        "pos_names": ("i", "j"),
        "pos_units": "grid steps of the area's reference-point grid (spacing not stated by the source)",
        "csi_axes": ("rx", "tx", "subcarrier"),
        "device": "Intel 5300 NIC, Linux 802.11n CSI Tool",
        "buildings": (0, 1, 2, 3),
        "license": "MIT",
        "doi": "10.1109/TMC.2021.3073005",
        "citation": "Zhu, Qiu, Qu, Zhou, Atiquzzaman, Wu, BLS-Location: A Wireless Fingerprint Localization "
                    "Algorithm Based on Broad Learning, IEEE TMC 22(1), 2023",
        "url": "https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization",
    }
    areas = {"lab": (0, "Lab Dataset/"), "meeting": (1, "Meeting Room Dataset/"),
             "conference": (2, "Conference Room/"), "minilab": (3, "miniLab/")}

    def __init__(self, root=None, *, download: bool = False, verify: bool = True, area="all", packets=None):
        super().__init__(root, download=download, verify=verify)
        wanted = [area] if isinstance(area, str) else list(area)
        wanted = list(self.areas) if "all" in wanted else [str(a).lower() for a in wanted]
        if not wanted or set(wanted) - set(self.areas):
            raise ValueError(f"unknown area {area!r}; choose from {sorted(self.areas)} or 'all'")
        if packets is not None and (isinstance(packets, bool) or int(packets) != packets or packets < 1):
            raise ValueError(f"packets must be None or a positive int, got {packets!r}")
        self.area = tuple(a for a in self.areas if a in wanted)
        self.packets = None if packets is None else int(packets)
        folders = tuple(self.areas[a][1] for a in self.area)
        self.files = {"all": tuple(e for e in type(self).files["all"] if e[0].startswith(folders))}

    def _parse(self, paths, split):
        loadmat = requires("scipy.io", "datasets").loadmat
        paths = paths if isinstance(paths, list) else [paths]
        folder_of = {folder: (area, code) for area, (code, folder) in self.areas.items()}
        X, pos, building, point, packet, ids = [], [], [], [], [], []
        for (rel, _), path in zip(self._entries(split), paths):
            area, code = next(v for f, v in folder_of.items() if rel.startswith(f))
            amp = loadmat(path, variable_names=["myData"])["myData"]  # (rx, subcarrier, packets)
            if amp.ndim != 3 or amp.shape[:2] != (3, 30):
                raise ValueError(f"{rel}: myData has shape {amp.shape}, expected (3, 30, packets)")
            amp = amp[:, :, :self.packets].astype(np.float32)
            amp[np.isneginf(amp)] = np.nan  # 20 log10(0): the NIC reported a zero amplitude
            n = amp.shape[2]
            X.append(np.transpose(amp, (2, 0, 1))[:, :, None, :])
            i, j = grid_position(path.name)
            pos.append(np.tile([float(i), float(j)], (n, 1)))
            building.append(np.full(n, code, dtype=np.int64))
            point += [f"{area}/{i}{j:02d}"] * n
            packet.append(np.arange(n, dtype=np.int64))
            ids += [f"{area}-{i}{j:02d}-{k:04d}" for k in range(n)]
        return SampleTable(np.concatenate(X), np.concatenate(pos), building=np.concatenate(building),
                           groups={"point": np.array(point), "packet": np.concatenate(packet)}, ids=np.array(ids),
                           meta={"buildings": tuple(self.areas[a][0] for a in self.area),
                                 "building_names": {self.areas[a][0]: a for a in self.area},
                                 "packets_per_point": self.packets})


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


# sha256 of every reference-point file at _COMMIT, computed while streaming it from GitHub and checked
# against GitHub's git blob hash; the imaginary_part folders are not used (see the class docstring).
_MANIFEST = """
Conference Room/coordinate 1-100/
1001.mat a2acd8d6371b55457eae6741ae686c3c98b3bcacc6cea4afc056e945796528df
1002.mat 35ccbe1def261d548ef4a0b0aa7ed7292d7dde9555fc61172ea6be756323afa3
1003.mat 89be16728718cea0ee4872dec78b6b6c69102de7fb1deb1c97e60c24a7887f66
1004.mat ae70f43c05d8963faae95252fa7085a6fd54749b09bf5cd78e65d95246d19050
1005.mat 1eb38937a904e4182e7030bd43f1087e105a9ae36e6ac97b0ba1064a674e48a6
1006.mat 87aa0ba0040ededda02063574cdc655693665a773b229971ddc0df3e9c86512a
1007.mat 00ed86a652020c6f9f88db90f24a8a7fd83017c9caee1ee77065ca8e895704b1
1008.mat b39b78e21f0b74a43e95fd5dcd4061ac660ef030847e0cf18cd4433c207b4772
1009.mat 600970389f3a526f3f275efadccb470e085cbc182d535f61bbfd6ebd1e93b52a
101.mat d1cdf9768e2bff43b2922e3a78f10720ffad765ede856f5c64b714263806cf25
1010.mat 1ae82136e71d7b75f52b24ac4aa63ad7448779ba29761029e008362ef29502de
102.mat 0ae029d3bca0b4f4e33c71763b00157f853de89bf360579b057581189dc8924d
103.mat 83bef7b6026df2b09418cd0d37d3e7ea26b4de9445c609221216c19e35869227
104.mat 501f44d255c90f6e6fb8785dca981fd13bd91c5442e8af4bae10d9c58391cfef
105.mat d0cb046aca933f6093633ba5e480b797954056f7db19a2bd1aa25ea67ae2e1cf
106.mat fb7f7affbcb8ffc2acbcea1296baa750b40b0e64727f5673e8bb2a7ce9a08e58
107.mat ab4c92d06f19020c69a5e039a10b666ce2637384eafb8d76b4776d2faf016a93
108.mat 6fbad65e857c33e8311d89822f70d0654fcf68a58241885505b283c351e721b3
109.mat 8a8fce4a690dc4d908f7a1b3b470f7ec79f1ba1e57f2509e0b3c106f017ad580
110.mat c86d796a5e9183664f29de2b6bf02eb463bee9758cd87d59a1093d60337b98bc
201.mat 2105769e96341fcbc03c6baf5c2e35e1366cb65b5b38043d0fb20a2ed3d5ceb1
202.mat 4271a88513022301d897853fe88d03b9b7432e6adc6300d0604fcc7077d36a96
203.mat ef47ff690b61c653018b06eef8944a9f3f7bfbf47b51151fa4f038a641c6e3a6
204.mat 25be3c37120000e296c33b74c06ed43bc90c3d54984616e070171709fd997961
205.mat 124753349f35ef6fdf66346132dffae45bd31c4de6e642955a0c56768358c68b
206.mat 5f661b5bb8f98289284ab2f7e57f75d88f6d6715aec47ccbacf9d53ae03560bd
207.mat bd33f1507ea3036a962555cd62beacc7fdf689ee3dd7d1b9ece3365d36bcc674
208.mat 604129836ad082beee8450ad677e52c1c025a2ec685cb70cd21fb278c721a71d
209.mat dcce4b9961b9e7438d8a76b2c68b92a99340976a30626208ca2b5a13a5cbba25
210.mat 322e9c0c94440565145c94ea91f156cc7b3a465a1ee43b9af7791785869ab04f
301.mat 3db29a5ac7e3197bf8a9e964d57aa711d75ce3345b0fffc15887d01e7b4a5ea0
302.mat a3a3acc4d38eff17b7f46d52052a3a55d424b0227c3a030333a14cbee2047fd0
303.mat b5b405ce0e0bdae5d92672bd2558781ae72f2b9f69a1d9f1d5cecdbe659ad544
304.mat c220a3bfd7ee286a3318269e1f7431a16acb03a0fb616e5b8ff9516ed1e4569e
305.mat aba930e5eac12e5e077cdec5368550649141949bb6fbe993d320f27caee92c23
306.mat 3b0101f03ce9517ead9cc76bda98af10c6756f63b57b08ba62c4cf27dc8fb1cf
307.mat 302d60d54645e7b43bb9f3410498c5cff0373ab5d9f8ffc8063d1ec8b051721b
308.mat 8af61501d35895d47a68ab966f3fde197b1f8b97cdb7ce8e4ef8a520597ed681
309.mat 25f195044fc61b89987684697cd8959a6907371149527b0fc729df2302a9cef0
310.mat 51cb5a66957f1917ce0dd2857db5eb647100e2aee2f01f7ef862e3a79d83b007
401.mat f70ee436e7992fff7465dbcb55989ed59dad2a86a83425e265a3fc3598f8ac7d
402.mat 3146ecd8987570fbf2b86753aab2e6f91aee2e9b71d474874c5c4c2c3ea0135c
403.mat 354d41d9f84766d4e705e7705f1be401b0cec54544b812bd1fe2499aa2ee840e
404.mat 39e9d060f71a6870e35297a70b9043a3689d55a241464af5fbfec953a1136393
405.mat 7b565caeb951fba24551948ee4e5abcbfa9be22db687d2d9503226b54bb69fa2
406.mat ba3a7a1f533c5606fedbda2f2e394e56cf98a2c019b8168a2e7f82a7ca9073eb
407.mat 5391084bdf2681abba00daa1f0d08fbca17d49e246a323b8a525c0ff3eaf32a5
408.mat b94beff4953e1644bd059f291a4b8e959645197bc9f1f5779cb9e3befac13bea
409.mat 9a3bdfa8b6baee58361a10f293eb567c70bc824fa3822d394176d1857bb7c2d4
410.mat b8b630d6e398608396230f231ad0ddc2fabd6144af1d5d4ea78680c070ac6a41
501.mat 6a675d366b5db4facbaaf8f7efc9778435de531935e9549cdc2d19229b77222e
502.mat 1af2abbc6228fd454a89dfb396cafef8367a686881b8f99381a029aa56a94ea0
503.mat 5c5aa74db29852786cc8419e13bacfc334c7b9a406e89135198121494fdc929b
504.mat 5e5a9b808fb45b5555b785e47da38b751def60d75f21187e773e72aa1c8d41d2
505.mat 0f35840a5589b41c3b4756dc5e64b6bf3b4b3bdab904a7d32e6c7251242f0d1b
506.mat 3422718e67429676ce34ad7e8f0b547e1e46268ca585e5e528e4dc4da9415488
507.mat a1d722a0e686a31ecf69e2bf091344db76a4683ce7cc52f615411035e75dc70a
508.mat 650800269b9bfe04a1df4c9581a876a7d8e40be8cc2c61af416cf72834ba879f
509.mat a67cac570b909a1ed2e5a59d58c596e49394988f74101ef9dc6713d9e9b98064
510.mat 8eb089f287df4db6dcc5f352070a453c1d4286fd1c2825b40f61bba24538cbbf
601.mat cbba5a08ce89e16600689088eed20ee1bc5d7e53f1769eaff7a9be942b2995c9
602.mat 157af6c426b1f6b5d27882cf4c858873ff5f2e3b6ed69ec69fa33a2277c03c86
603.mat 7cffe07981000c3e45adfd4997da08b54c8b262789492a70c10cef2ba95ec618
604.mat 43c675152a2cb09fdd81cf4075fb7ec87bcbe57df6ed9d8d3b68b3b14aba79d8
605.mat 611ecb4d6d278f7aca6c40db2aac223bf22992f3a822f3a6e8a2c58c55aa949c
606.mat 6a3e6f1b1953f56ee1024a865a57c08f4d4f325a578a5dc946fe74aea243ec60
607.mat 2575113bb9c779cd9aa8e0c22230ee0592abccea43824423510efdb0cb6142e0
608.mat d5b68e0f99b90cbb14a74e420d70dbf5f0b083fab4c47cdd8aa95aa59775ceda
609.mat 6b47872844df6902cbf1f3007303286d3ad1a648829e0c2dc328de0c412de98e
610.mat 7683e13ce466ea381bdf812dd91d14164928de43a6df78d0dbd02f4d0cb2e733
701.mat 4c5504c4da22bbc066bc1d5f090c6f8671bcfcea257cea02fad2d022bcc0e68f
702.mat 883cb0dfc6aa9217e9d6222f9370f84086ad8e5d8a98a19e4441c52f67b9cc8f
703.mat 87e56c158bd1a7d3365d0ec3d772236badd102b3e5a911c0377b60739d4e1122
704.mat 13d10c949e882dbd7c3a063a632683b768edbaa5d56d04cedcb58c0270eddc87
705.mat 904c98988a080e228eab9ca52e01c3f31125d0be0ae4cf448e85b96639b557e3
706.mat be8d8d0341b17702467cd60466747916f3bdacdacfc855c48a50d7b87b9703de
707.mat 4f99cff63c03e10693c90191741bb543ead581906c10c59b04aa721a6ef2b452
708.mat 2b2384d754a56b23c824e8c26bd631425da26a03229571cb333131e4dad08baf
709.mat 5314268231eb694b85dcae02b8477a5025f82b939086d973c666243bf960fd5c
710.mat d61e55094139081e12a75e373bd8759c070e75a469e8e8de646d8d4a6643c2b0
801.mat 673863f51a2ebed6a3ec6bc5444cca503542a869602742bdb687fa5dacdc79e6
802.mat aa29c2ed50448c09aff0f9e1a0281191610894b962fade59b1111b3998ffec4b
803.mat 149e8f272a63b955f180aa6df95ffff3f8be6f8cd0d3c65bf9195c519a558d62
804.mat b2596340cc3dc73cf2bb45087a792a787c4f6bb4939397cc78251bcef936289b
805.mat a02624df00115bb214f60c279f238a2c5ddb15abf4e86d4dbd0a32e0f25c9b44
806.mat 5cc4c7780e22ed640009d38b981563ea6c579b05f01695c41eb5fcb065987082
807.mat b0531d57bc4cd7a5ccc6f6f69bbeb5f8de45e9423829a1f81177e5ec3d37578c
808.mat 0dfc4059cf7a11e65adab9b8bd2dfb92f61aecced136be05fd169a4a61151f8e
809.mat baa4de80edbbbb5ce159fa2ce92d8ad4b747bd9542425dd5fb9e677003f0de28
810.mat 9d1eb2ed2201be2ad5418af2e2e08cf3e9ae15206e12678fc7edc6bc7af5254c
901.mat 515b002fbae0d74ab7c448338c62f71d4a6f6899ee76042dc2883737ddd4e46c
902.mat 7eb3312947931d9b06d4e6d33d0808465d6d03d5d28ba55933e24900a916d1dd
903.mat b5898a591c979f3da6d3b098ac3a870d8c5d88d87cd55b090b562f45d38f3b70
904.mat 2d615e8f7d339d62429622b975d1e047f0a0d3629657f32d7f1fd47717febaf4
905.mat 2420b68fa3f7db81142432a9333366f0fd2d52670667f50bc6aaf3eb3aa50eb9
906.mat cfa60f757b2a3932b3d3d13a45228857f13fd1736fa35768fab08cd967741209
907.mat 7ecfcb6935f6f55ef364ff0bc9717a84747554148b515bd14163daed367b36e5
908.mat 8a7c06d2dc9c16aaf3687a8e0221d0ff4b9d31c66133189e5d13e3e382740a4c
909.mat 56552d1e275acf89df580e8046dcf42022740c5d1e31cd3eb1dee1408905e273
910.mat 243b2605148c68028a898e07b238f9a4c4533b8d1ee8293420a10bd24e7c1dcb
Conference Room/coordinate 101-160/
1101.mat 19fa5aac90aaa0fc53484f229e5e851cbe54c5f463b1e8bb1f2a235a5f8c5c13
1102.mat 626db0389e4036c072743d9af7d41fcdf7bf4d0cb8fe6c5825726870c0f4c978
1103.mat fe30d77a09f2613ece3040aa3de2244d4e3757cabc36438f5c72ca7f6630400a
1104.mat 396813a8f3c46680e71dac3c47dd2cf7fc5a87310c0fff7acc00893a869cec28
1105.mat a8f5ffba2395d99fca072d1c8a0b472db441c6a42be3ecf82250c614e6c6b84e
1106.mat 8b89283726d9a13ed59ff69c7432b13d1cea2f531f329ea9f4f23fe8bdf0d88d
1107.mat 5a492b5a56f968b82344a465ed369057ecfcc61cc980d7007d6eba5d34f4e9b2
1108.mat 4c13f665587d1cc4c9440fbd26b1360d17063035783643d0dd6dc6ad01dbd9c1
1109.mat 7e19296fd71794a06bdbbe8bf79585e2a7869b172f70881c18d094bef375b583
1110.mat 43e4ce7008e9b52835dd5b2519472659838e56baec643dca59b22975a9fc1564
1201.mat 916699b7e67b4846a7a7f135decd5e3e319ebea68c7b42864fe4d7f28879c39d
1202.mat 026c219faed4b2fb80c9a590e2d1f80f21f506e441bdf8faa61b8a2c8655c992
1203.mat 4a884e34f22115e000fe2c926df6699ccb1b539b5a21d6b1d7905bf9cc0a713c
1204.mat b855f1d4a1c0a8d33a4b17d184a17aab9d65ecbf6ed2ce4b6917fc1587da1e76
1205.mat e4cec75a42deb42eb58e46a42a36ef65d26d968a8822b26225ee61e9ebb1f75c
1206.mat 671067fb06eceee2893a53731cb62ebd8a19987341819475974dddec80df1c4e
1207.mat a1ed7d7cfe78d89b23bac49db289563955683df2be42371cfbd57864203a419a
1208.mat 07eb01597ceef8a3854c9eb8ec995e548060fa90700ec968ddc60a28996e2f87
1209.mat 926a14b31a48c3628727c8647f2f4e501ed23bf6820fa022eadbbe6939070381
1210.mat c213c6403fb753d76f34b89595a83170d37d1d367ab836e61cfa353837579e73
1301.mat 1b58e7893b987a1d739b92511c15eca3f1a00277df067f4f6e14d28d160e4f90
1302.mat 67c93b0321f3a325f2e8f481fef5c641c33ef14e79f7bbc3a89ed6abd94157be
1303.mat d577cba78786f80307e4930a81700373d9f76c68c2a08ac53b5738f7962dfc6a
1304.mat 6d8a72304637b430a48c9c83fea9fed234ba3aad5648dfc8fd1496ed4b747a87
1305.mat 94bb3bb3d73ac7beb54579455c53acf2f8e8dd65c777096d569c1a745ef6390d
1306.mat 416e4d5fc1f7b75d5a5b1b2df7ce73fed9c2df164aeb215c19b4de5c06ec490d
1307.mat d32ffc1266b52a031dff448a67c3b656cdc56b254f806ccf14aa4c122dc2a01c
1308.mat 3a8008f534f6ead43e43013f4a004b1501305784686c0b4096e5ab5d0676417c
1309.mat 06dafb37dda28e7010dfa7802568f3a2785b64e1d6e89a27b20c59dced0757e2
1310.mat 249bd18264f723a05eb83a30c7b12756d493ef7ff80214a12804e6d794dcc2a6
1401.mat 5ac24698ae8ef4b9f2b42031331895bcefcf76a57546cb6f15bdcdaa25c44ee1
1402.mat 48b8bf6586d3e0d760968f11448c06ba18c10992d097e38173dda01ca617d247
1403.mat 746f8536c5a8f0e01993cd718a7d01ae32843071a39f39c59b9233d0f890a564
1404.mat c79d2ccc27be2f8e6ebced0994cc53cfc29faea5d66c323f3613e5c3c773824b
1405.mat 47a6ea5ecafd2593f71d19efd2120e606d51c6eb07f7192f06411a85b7b50db0
1406.mat fb36aa2a4b3ef98d19d4f103226974e797d514fe4644d6168563e95b6f925d71
1407.mat 07eeecc0f1c31e90c8754c8bf856685526dcf5b091c2f502f9e8cae4b3914cec
1408.mat 8868303abc3fa922fc39c43bed2978199143e228591da54f27d9a2a2724b84ce
1409.mat b458be245a721236432770d1f653612848039bdbb29e76aa1b71f5e5a145b11e
1410.mat 4c535f87b9c5f6f259d2fa5000a9d05af5c455567c3e0b745730128442576003
1501.mat 80e4c7c3ae61119b7204aa88bb1316c87e131c10e5df9353a491de1330ddc667
1502.mat 043467f21e3010f6e1ce50b5bb8d2f08ab646d838b8bb147477b76e01b569dfd
1503.mat c0974c3cc78451287a29fcec8b0381728d460cc491c83ba71047f8f6923a4da1
1504.mat 14bcab97b933f6e0da86413ddd38675340213cfaef2464a4b6611cc658168ec7
1505.mat 44bd007bbd13c37ac32b707f200f3a3769a673526287b3dd21a9be3f998e643e
1506.mat 2286aa74b287b38938adbb2bab15e92b65f5d64fb86fb6061802df7ad8a09a8a
1507.mat 56b8819f2aad29a2ac810db25e2935adfaa67d2cb9adcc43752c69e518620c4b
1508.mat 38926dafd69866e2315eea4021f064c5dd67fab86591c7b787134c9fedbe5a9d
1509.mat 4d4beaf3834bc136e4a0a0f06bf52459054c0f5e5bfd9a9bfdf49a2caed3040d
1510.mat ba8d318f1ea12e49b4df3b235b1067eb5c8fc001e78e329291b2e2d6d8846687
1601.mat 48329042010801e69f83dfa2a27ca7c106ca546b1182c943152a3bd61857309f
1602.mat ec754980208ee445bc54742ddce24e9c9b7d59e3a8b6793014f7592bcf12b11f
1603.mat b44665d84c33f392b5e592c778a92a5cfccd689af6d28deba74bf4678ad820a1
1604.mat 5e4b2cdace86637030c3188498a3cc323045345248d06cb3bac25bbd54179269
1605.mat 6b3442d1aa0b38ae5dd9c6a938a4140e864fba9f7a691e03b634b4280ef817ab
1606.mat d472f4ac16cfdde29fb222ce7c51ecdbeda8d204fb3ca840d7e94e0c59e7f374
1607.mat 99fdc66e5253b29e69debfed175746fa04a4a94bd3423df55dacb6b10a03244b
1608.mat ea5e5edcd30a5f98ca6adee8d27dad67ebc51e0a0afbf522ae5ee564740bb8fb
1609.mat e2e63b49465e7bd876c829273b61eb01c0170c6b7884307b21c4075eefc342e2
1610.mat 92b3ea68c580495a699eb00b98cb41af1aec509c9608d2fe456bf20ebb4365a7
Lab Dataset/coordinate 1-100/
coordinate101.mat df07a013de2079e5488a5419ad57b4d4eaf319546b402ef284048476307cc3bd
coordinate102.mat db1dadb22919b35951c412f774d8e792c214c7d22a140d108353ef4b243b720c
coordinate103.mat 6f4f9c7c07c80d16dbd1c42bae41dd285aac28251540e7187c6a304732f188c1
coordinate104.mat 600dc3d62a41f2f6684b4d0967d7c381b88a36394098d9df1c3ca10828589331
coordinate105.mat c57a9292fc01706a14f227ca9af36cb7675d02be8ca217ac82ce99678f0f7855
coordinate106.mat 64c15c7012e61e0f175a1c5e5cee8a737ec68fdd1fdff2889de8939959306713
coordinate107.mat 2d84208a85c8cd6af12e584c4afb1ea47050b1ac8710352e3ab73b2126b0ae71
coordinate108.mat 31122672dfb1d1d875ba268373e3ac1701051286ec2d62903191a7f1a283b52c
coordinate109.mat 5bae1bfaefca7d841e34a098e8953691168e5500f1949643818bac874eae36f0
coordinate110.mat b1fb485a3884115194df9d56b645f1256a7567b12d8d48508b1c023e76533dbc
coordinate111.mat 8d045d47529099af039eefcb1c1eb59485edbce31f1c5d57f5802f37c91b5401
coordinate112.mat a16b6700020ce2cf77148edbe055cf8a6a222483010fab64b5388dddc294f71b
coordinate113.mat ad26888a8f781670baa8a262979119e74ddad2a76713a3ffc091c898dab472a2
coordinate114.mat 79f85922d35809f13674e5332e0a3a5acc291dc9f83c118d5aeaf6de878b89a9
coordinate115.mat 23e7c137b31f1b78b438e932bb759d4508129e4a7cce123e9f835e70baf022ef
coordinate116.mat d94b9bfd7acd404cf6a31817a0117da2919dd0bd3d52cf0107b33dc787e2549a
coordinate117.mat 766c28bd9eaf11ec22918c08bf5c95935249ec3106efd4cc558dd24f03188bf4
coordinate118.mat 05ce0849bf7fa4c2f9d7ce39bc247aca71c78f0cbfb3466c821495a8481e869d
coordinate119.mat 1b2f83a63bdc5c4ec2c6e2f4e1f37c2f88a0e23c222911bd7f6ca26d72a1bf1e
coordinate120.mat 99e008d18911ea87fa54629b13171e4d7a1581b1d02143b395f05e66b448c600
coordinate121.mat 9fae63f8a2d6dcdc3ea1e62f69d44c5b4277e6ce995a981138dc4d078683d208
coordinate122.mat be4b71493e7c6c820f45d730a27e63f01a6119cfde1987cb47227e9042e39ad8
coordinate123.mat c355131a19e209ab928f2778460638682ff6f2a281d85ae104ceda95b9d5c9df
coordinate201.mat ebb6f27d8c09f52fbb1257061821e47757ca9cfc9330fb76e6aa4f2ca30b4561
coordinate202.mat 61644c6e7cc164875cc056c79cf21dfb2d50855d097ea97702a55975db784bbb
coordinate203.mat a21087d72bdce452705dd6994bbb7a4575e86b9bdce4e10b6da474de202d70da
coordinate204.mat baa7066b20129fa3f088b0bd14994c6601a4b6fbeb3e9d714d8d5b906c41ed78
coordinate205.mat ba11b413b25c19196e62f9fed5b8914510a932c510d91d888342189d14928f21
coordinate206.mat a6bcffdbce2d2423626d5b1b3e7b6f9d214da3cc796e457828a8edb5331c84de
coordinate207.mat 75bc3803f06371fa3002dd7be4417e32d19ac53810f9b16378acc5deff3e2503
coordinate208.mat 594ded6c7d772e743da153568262b42bda8b1bc5521c1fa0ad289e2095c74af2
coordinate209.mat 2a5593b8566f9a868ac90febee812a2f7e28d71c2a316efb0c002bc6241ea74c
coordinate210.mat 67bfe83a88de7fabdef7023b3437f0aaa7817c108230f6c7f26c87e15b7fd055
coordinate211.mat 67a7aa6ea6efd07211e0ae12f86411b1ad5bbd63d75e05d03cc0494c05c2f45f
coordinate212.mat 8a44b62838cc2df05b0461cb622017aa04067602c0ab5829612adc7acb037947
coordinate213.mat efa5825b167ce627be86fd3a24045631c2992c12c0051fa9c78a0fc5db0a3ec5
coordinate214.mat 27a7b847208a23a1e61b8c5f3c4d8d3b3b4a19dce784d056a283ba468af00bdc
coordinate215.mat f627e3bec14094b7f3b2206ce443eda2f08c819a34e0c0cb3af9d77c30c07756
coordinate216.mat 2dd98220ee3ee15ea5b84788e04cedb4d11a55d26526ee7ef0e4638825f10504
coordinate217.mat f7962cbf6a8ee99b2b7a07562faa2cdd1ba181428febca98ae698fdf5519a73d
coordinate218.mat 0bc4d213960000732d5d3df386b0d2d0e96ae8ab450c0846212e63098bb3d6b4
coordinate219.mat 5404431bc061fe7c01c0c53d4fda0026a236281a41571f4567809eb55bff992b
coordinate220.mat 1985dbd83de90727e5dbd01cd9355b73483080a2f2f59f41cb13f059dbd25908
coordinate221.mat 71a70da938f598ff759141a2a97b6307c980a8a16945d901731cdb1d9cbaa66e
coordinate222.mat 89499ccebc2a0b20b80f8c905726b37ad07479aac3c56b5f7a685acda1b08d56
coordinate223.mat f4a8e332eca75e13e7fd42ffc0aa53b3e7345f6c7024f324b0acd271687e1fd8
coordinate306.mat 7561ed26915fd2930383b9b4d6e502e2a2447ff6dbb64a195644a5591568fd59
coordinate307.mat 9c2417ca4a10c1a8b49b35edc07fc48ba40289da0b97aa6a972788dc74a43b2c
coordinate308.mat 325082e98a459ceba231a086c84763c16a16dac92dd4a9e9d98e870974aa32a8
coordinate309.mat 9e219a81bc54ccb3e3d04efca1fe00e613524be58d1f3a06475d435ebb0f5451
coordinate310.mat 50112a19afc63cd62cc82ec8606df66f564bf89280c6a1d9e4615e5ee274373a
coordinate311.mat 7398b20d6410f25f7f3e4fe37a875c26d34c79265b4f53966712d635fa9102b9
coordinate312.mat 29d345447c4527914790dfe4fcb07b27227345e042c2071329487b5311602319
coordinate322.mat c35dd68adb6337ea51551b95468099858c58438e22506f82b5b3c9d785490634
coordinate323.mat 2787b987e4cb8533ea585dd2b6940650b0e683d869543b168a3a4855804b1baa
coordinate406.mat cd1ecb0bc5de3619411d958951e16c4a38cbe239cd82985e2a9c66a121605dbf
coordinate407.mat c0041c2384b6935143951bfc44ffcc1df6d7d3a7aa4a292e029b9dcbc446fd0a
coordinate408.mat ffe4692c6b71a32cedc86ff45bc60dcf9dbccab8e0f88f494ad2a31809b34099
coordinate409.mat e69dff4ce68d16e096e5bf35f3f9ce75d83c55f8c524c93ca9659796bd40c12e
coordinate410.mat 3064961ce8885bbfcfe4ff260c175837c3becabdfb94ef13871a932aea4a39ba
coordinate411.mat a850630ad3241925e1206c05d38d7f1b110837351060bc33fc2f3f01451f11de
coordinate412.mat cc99a5b5541110900e6b9984ed289fff2d191250ea2e2bc2aa99e7aa9aeda184
coordinate422.mat 88658407f9e87731b6636494a33263d27245441999918c30a98a6630fa337783
coordinate423.mat f89d44d428669d06a57a0a18bcc305607cbb544a6ca3d9ea542f753b3441cba2
coordinate506.mat 7c98725e667d742c042038fd23d5c2406e4d2230dcf96ca83444daed520dbe3e
coordinate507.mat 69456fc3238bdc98a6ae001b86174e48870606cc3167e6907c1a296f613c15c7
coordinate508.mat 798d75e638b456d43d11c741165fe5aaf7ba49ea08f534ecfdb01b670d99fdf4
coordinate509.mat 3fbbcaad85b1128ca01301bae38ace823100e7c26edb7ff33cb7fff657953a75
coordinate510.mat f4d4be6f22256a4f12acea7dfd0fe27fef23513e3aaa2995d0f625431540a8a2
coordinate511.mat e8d9f5c77b1c432ea08b892acdbf9684d6b02511e74f9167e735dfbd3433ed9d
coordinate512.mat fbd4862419b9ee13e11861fa8719fb9b1008504234fbfd85850636bee814c4f9
coordinate522.mat 6d7bb4d8ae7cdbbf363ba81957ecb3ae99260715ca1b0d56476a71d6641cbd33
coordinate523.mat 1a2cb735be48090e977c7667294f653e9cc1dd9aa34e9adb7318a4a6bf3a82ec
coordinate601.mat 42431a0c19e9083979518ae8ba6bff153f78ea8416dff2c585ea7939b270f43f
coordinate602.mat 332ad2447a56bbe82dff2385d0c56278d246acd00ef6ee48df3f403580e2f288
coordinate603.mat fa10e842bcf757c51bf7d9da0626476e76eb8bee27d1cd531561b7f335a23b80
coordinate604.mat 802b3193267ac6d4f7b331405cf95788ee2998f328a4e4e2620cb39f702eb6d5
coordinate605.mat 80d32bf3de50c97512fa356dd5e3f5098d6ab13a27c96d25956794031a9e7c16
coordinate606.mat 9a2a12eecc27395f60e9e20d0de2dd566480157bdb144c11b8d868466696fc51
coordinate607.mat 0b3354da7930df45df29e73d7b7328c176c006b0bbe6a81c2e03e7817ee25711
coordinate608.mat 7ba92de59cbb908644c44bbd30b614dd07ddf2e3733fdd77a88e7a73bbbd1a2d
coordinate609.mat 11c3f1e740ff0ffe15d7248402618df01577114862b3a635a5fe513a81b59e4a
coordinate610.mat 04f00c42f548e573fb3e2227d7528fc9bfd3ef506c3b5deb1f8cfa5336742f6b
coordinate611.mat 814fc972819a5cee056502b7f71d1a8bc5f2646f03569367462a78e804d2bd46
coordinate612.mat 25783f41b3c78a76207b6af4901b55e6ef8509fac6bb583384abf43013e8ff30
coordinate613.mat c6b4f5ebec4d72fdec72877b327fd122dafb21cec3cf32e43dcd215d8b1430aa
coordinate614.mat 382a2d507f2cea81f8c5d260b01c365b6b3cd5847031bcaca26480c0e8e58a61
coordinate615.mat 484afe0f9325df324b14bde62fff9e03c37fc8ced70a664a49def3fbe0183231
coordinate616.mat 07383fd7f17af92f5b8e1ae3afe7c07fbfd899b90bb971834a330a338a6704d9
coordinate617.mat 71c97ec529ac81c995124957b49027761929ad6b027af78fbe0861075f0ccfbc
coordinate618.mat 7959ca7e4f1f544381b293af68973345787eafe0014d6dc583dff70781a335f9
coordinate619.mat 77e84e0aa5ea073347acd91e8264cc37bfd1c272c6dd3d8e70513992ccfcee7d
coordinate620.mat dc3a9deaa5f42ccf6dc3c6a2fa835b43c17eb322001d83190edbd39cf8c8a84d
coordinate621.mat 27c5899414f8555ae2c74bc1b68798c91eecfc6d46c1c501691efc7585946a22
coordinate622.mat e29bfcd0684c9855258a4be81cc0355d994833b31ecfbbaf581581f50b5e1f2f
coordinate623.mat 58da82db7b0b807033022198a43f6390b4a55a7d15e9a216c758d09c8a1e9610
coordinate701.mat 230b0b48d9abefd6c79787637fbf2cdc8da0b3fa45a1f1147151bc9065a1814f
coordinate702.mat d1c358aff9d5b5a03b48d7dd4a2600c29a0e017abfe67f8c11b323af7fcb88bf
coordinate703.mat d9e10ca5be65bf8fe5214760f8468cdfc72afe6b0b44883c2e9da9cacc637d3a
coordinate704.mat a75d87bd46b08a0a38b2ec1bee3baea24b806ba61b5dc43abf8d4e6bb9d6fe56
Lab Dataset/coordinate 101-200/
coordinate1006.mat 3967e11aefd5ae6523c6847430d42ab0eb93ccf84deb0ac8ff5e5afb378d99b7
coordinate1007.mat 42f39e1c64b587b54dd2a02d5f9b16b74edc429a1e31d63f6c98a5af3c8212ce
coordinate1010.mat 18e98fcbe3f8d79782d2a80fd7bc02939ac7620074e668f774b6d43033b52275
coordinate1011.mat df13af908b9af914503aebcb0421deba1edfa0441609724214a0232eea326195
coordinate1012.mat 1edacc0a295ff9def248440204399efbbf484770272253f716f8c1583b3b296f
coordinate1022.mat 93915c37b0ca60aeb8bcbe15885aa8355c17ccfa110e6f6325927e9264491b59
coordinate1023.mat 06b06e7b428f05cd09845594f7ee9080fa32abe0cb6387f0e8baffd6f75ba1ec
coordinate1106.mat 67336cd8a6fa85b0733347b539e8d384cf725e2e5ed9b040c054e38e5aab1d1d
coordinate1107.mat 9377ad8e3789849aeb284a19a3281773f7f4c6468179157a204a7c2311321cfb
coordinate1110.mat 55466b5e30c7bcf40a478781b8606f461ebfc6377b0655ad66406a625440ae13
coordinate1111.mat 6a5958848198a6b5e3a865279a08f948158e8177d75abf828024a15e771380da
coordinate1112.mat 8cb1e95f24ad6982ed5211c83b64e63b0856663dd50a27ec2aa4504eb0ff882b
coordinate1122.mat f58c2d809460e7eb5b722d6708ff0796b4f260efa67bbf18b35a958af6b90828
coordinate1123.mat ca9ce9f69ed97f0d7ae8e84af5f7abd13e9bea613530f25783a379079d48f026
coordinate1206.mat 4a61d397d50b82e6e533d8d13ea6741ebfed489f74ab2c95b6e1b31d496b1659
coordinate1207.mat b201028998bece3cb6fd7aac61a282faeb00ac1b53f1946545d7d05df2027eb4
coordinate1208.mat 8ce3be81fef97a66f4116333ece7c8a652103408faad379167def12ea54ffe17
coordinate1209.mat 5d611e1c72c484aa39fc811f44dba0760cf3b7c79fe28bbc95b3d98b5379108f
coordinate1210.mat 06c5794920b648f4b0244b24cc5e79bccf70d85c87a399d70a3db6bf6dfcb344
coordinate1211.mat 62da9e90888d09e40ae50e16e022a50afe5c4282090aca617d956fa3beb4a02d
coordinate1212.mat db7f090448503b13a959af7e06dc3d670bcaa1dda173928a5d7160d28b4a8822
coordinate1222.mat d232476b3b553367c413e7862df46e26788ce6b4f4def21673d8f4282449c8d6
coordinate1223.mat 0871753c8d74559933d41d8fde23622845477c78a023d2804dfe136a9b8be4b5
coordinate1301.mat a874c377f2ed07a5362d5fc0efb907fe56b04b964955fc350112d308fd3116c3
coordinate1302.mat 62252b6b598db258d3dc35163fec13116bc1730b4508dcf6b30d3ca25137bfad
coordinate1303.mat 8df46094a115bee58dc95d2c77305c45748568066d3c487a7628d9f64a7f4ac3
coordinate1304.mat b6277206345e82b10f1253c57c79f9885cf47ee2d6c0aa3efda0f1b6d1381b27
coordinate1305.mat bc81f1b3614cd36c0a05150b8f6f64d0004d400b53c74cab06e107b97a4fbd2c
coordinate1306.mat abf86948b696f85a2470c9d2b1be2daad54b66e77f1ce7f69d8328a1e1286c2a
coordinate1307.mat 1b451c6562fa03df82b5f5665c755f5ad45abd3ac7d937dfb8cec08b3e9c5097
coordinate1308.mat cfe13ec92e10645e8a309ae0a86cbe92190014ffddfd5e621df2eb8501165f2f
coordinate1309.mat 2fcd4ef26f374ec27d1e1b97f585cfba2c7b63554d18a7b487cf091ee2e6bcfb
coordinate1310.mat 28ecb66de034e93e9aba5a93b0e9a8b0e95aca3d773af16c93364f47216a26df
coordinate1311.mat 61da254171681815dbd70f97d897a703578082227681e65c6451e08354a452fd
coordinate1312.mat ad56a3c4aaaf60680b1d3e088190a5b7c08625ee77a92f481734c4ae31595b91
coordinate1313.mat e250e2fa2450fe8bf4cf72c14c3a0cef48c45fe305187a2198c866a84f7851a0
coordinate1314.mat 509056b43373c27090facf7c2f1b21cbb9faabc7870316c6e5e43cea19c24ab5
coordinate1315.mat d29cc9573337128af71855f8a3bd3ce80dd36be9cabf3fadb244722a484e1cb6
coordinate1316.mat 81b14389c2a7aa4654fc20a473eca704a1fff98869a112dccd3b4e88e0175fce
coordinate1317.mat 4f40dc9e2eacb1869c9a6ccc4f7284edadf4958ddd93e30912e027ee67dd9595
coordinate1318.mat d95657920998eba3a4d50fb070768aaed5ba9d42e49de0f82c757dd5b31120a5
coordinate1319.mat db16b3c901a7a289f4de41d349432d3b6ef34c7c546e7d778a369e617c857e55
coordinate1320.mat 2b98a8866783d6be71018f68fd2a84f78a6ca8e361f5c50220cd1a1e57812750
coordinate1321.mat ee24db2922f242663b32dd7c000216aa99791a2b36265eb8682cc5b92d49ee87
coordinate1322.mat d5e3d341b2d08fb4c242a417e0abdeb6ee517dcd6e2af13374189cd2d2787bcb
coordinate1323.mat 1cf4136e9dee48676f851ade57aa7dd8a02e452a75b0172eb7ecb96467b65f23
coordinate1401.mat 1f300122550840cc7903673d63ae67d081e6022a87850b35d93c9e3c6eef839c
coordinate1402.mat 233ea898e1c605c0af40e543107737864e145fe5b1e4cac367cf8160b3550ef7
coordinate1403.mat a3c5bc9c1e1d269bed4a5342c3e35139cdbb2c5d726c3881d66d7b6336c23025
coordinate705.mat 0099e9285c826636b481b411938100a60483f78037fe843ace5e4ca8d1523135
coordinate706.mat f768b09b2dcf23e51e090ce8a3eecc4be099c8fc4ebde9f630959922e59015aa
coordinate707.mat 5f3453e896f944485c757dafa22150354d2d2207a27b9647e172c1436c4eeef8
coordinate708.mat af8ed031dadb893c8a40de0edcf69f3c37a40994479e358c7314de09ba8d3f6a
coordinate709.mat e46fc9a82385706786936b7d700536a0e9a576efe536e2f3432cbd7a7d4010c0
coordinate710.mat 61c93bb4e9e3ec62e9e5770f5975912ffe24d2c3f833f9728a2a87fb0568678b
coordinate711.mat b67212f2394fef7551a54138c5b892b307b159ae4bbdec7778940443f7326199
coordinate712.mat c7ab9904832d20ed2a620442b8c188cecfb2929d08f6e5746b83bbed9bbf311b
coordinate713.mat 40810d3b62af36e155c51599d97a0506f57d66d2265dbbb1b8ab7e2ea9a56d8c
coordinate714.mat 140211d5ec7f65a79b62d5c70064990534856aae68b0f57192669fd5b2837a77
coordinate715.mat df4a46aa0890ebc9c836f498fdfa621ae460ecaa1116f4ec1f9530a110d9162e
coordinate716.mat 0d7301b6d10c10a479d997c69b3b81371574cd12e26276bb2b1339d2b2030c4d
coordinate717.mat c23160206de5fea943b85019d56e2db0b6bbb8cf945ead808f65bf27252a642e
coordinate718.mat 60af92b1cdf54f9aa9f1a7be9e20cd23882193a0b3cd4a8b5109537b1b90bf61
coordinate719.mat acd988715da73143912fb74f5e1609219ace45063bc5f26239ac6e5f963c426b
coordinate720.mat 28ecfff517bfe9c30dcd402dab540516da93542548ac04d7c89315b168c525e2
coordinate721.mat aebdfeb641efb0c1f4dab16c1328379ed2506d0ff0ea9f0bd5039af514c222fd
coordinate722.mat be897b4dfeae275f68c576d3c760ba7c00716aabfbe5eefaa8ba83e0886ee503
coordinate723.mat 574cd25da0a5d79f8ae75ae60d3a0ebd0274d5534b547325650da6599f87548b
coordinate801.mat 4bcb2dae4f69b5c3e5ac9ba29328ecf97fc27f3bb9dc8bfc9d429057eb1a49c7
coordinate802.mat 2b174eb6746eb34460f8ba024cb8ded7cc59f7fb972a0c06115f4799099a0fd6
coordinate803.mat e86b307ca58f3301a9f8557a7de0654aec443f782e001f0b0a862f9f54d6040a
coordinate804.mat df298721e32477c086d8dee004a60932cc125295db167461f26af327964aefec
coordinate805.mat 6ef90d6f0a6690aec288150bc922c9468dd5d62d64b31e2f8d9798e042659403
coordinate806.mat 007f8675c8fd6b69383f2fe491bed92a75b14f2fc1594a25627e387e3f96de57
coordinate807.mat d4e2c3ab8c6725d29847acfdd0b86fcc3aee23cca050c7e33b7f179cda7710da
coordinate808.mat 46d4e06eeb6824b97826d36f71eb988ffcd2002727b65386df11727f68b6dfd9
coordinate809.mat 3fec77ada578e6c056f4e70c9cdd06b73fd856ce75bc4087c69837b15eed1f72
coordinate810.mat f11dea97b62a2ba1802b3fb207318f7f6fbc29cbd08b9ee81af66e141e450c5d
coordinate811.mat 6acbbb223657106ea7d9692a10cd531f26ef6e704da4bb8a496332cfdf3deed9
coordinate812.mat 21dc8c075850e54dab3467569cd960467c076f87cb5295a4c587e5f802d0830d
coordinate813.mat 9eaf31455c08be88a4f793e1a0101e2897ad2c3569799e25dc43e79da5973c64
coordinate814.mat 904d05051b54edda0f4b810e2da14b7596639b91861256057208e9ae7caf6720
coordinate815.mat ee480be04afbc6f860a77626a46dcdf3f4bfc77d56f2e30bf2bc6904f9bb6dba
coordinate816.mat 6775f0a7d0470fba6abbe7ccb0434a7c19d8506fdec33b43641459ef84840484
coordinate817.mat 6a756c3c1656800b0a33f9b1f8c7c21344b3f31d9b1734fb045c9f6af2537479
coordinate818.mat 5227625d13d4affc185a392d29797588d94ebb2dc873efd30507f345872bf450
coordinate819.mat 7678715fecddf519493432f46968e10f7a66e31e464374420cbe4de7b58c24f7
coordinate820.mat d301efe40053c1f1ab01a6451c186f723946e696034d826f916b7fb7c87d96e1
coordinate821.mat d381d18d7765aa1a0cf3de1b983cd4c9337f25b61060e2470cb5f6adca593b5d
coordinate822.mat d828f1745eca6fe9efb7483696ddc2dc7a335c22bd788f3ec24439c54a9e97ba
coordinate823.mat 3a4d6fbb70d6445ca7830b73f353a6395316209253f30dde28c27fb83a985ba4
coordinate906.mat 2b5bdebf66e8099d49150276c14b018e8fa93657d323292f7997c74d58889a05
coordinate907.mat 087a31fad22176e9d1d7f8c29965e934f0564aa855ff5b4afdd31ac44c468076
coordinate908.mat 6bcb0010768a67ca48a49726ea2116cf46fbbbf57061a859d0b8c4719dddf7b1
coordinate909.mat 3b242ef5f046dd16787347a30e5ca1c8f06b3852070d947924bf081da6aab583
coordinate910.mat 5d3233194056b5704a4a9f06d460718061febdb3c0f2dd2cdbe086a24f7b9fd5
coordinate911.mat c95b3ac0a5b9882d8d1b63a58dcf97e5ba1d1ec898a41f71e8fcda301372c6d6
coordinate912.mat fdbc5a6fc2036e769b69e2fc4ee197f0b0d6e8fb88a0d928ca21a216925c6475
coordinate922.mat 762dc149f6d049f2e6d41e3effc91f6a3b599e443017af202628f9c76b05278e
coordinate923.mat 38d8d4cdbefe36390e5b4798d5926a7880c9bc79222d1e59a7010001fbf9cd4f
Lab Dataset/coordinate 201-300/
coordinate1404.mat c6b6b33aebc5d32bd60bf2ad3abea592207331d5bece4a35cd510bc720fa1453
coordinate1405.mat f7e4d253dfa9158fb202eb4a950f4750f8a5b363ddece6f7accfcb059e1776ae
coordinate1406.mat d14e95b26bf266cdb4e3c95dd0c359804b5fb51bca43dbb2a78a01320b2e4e47
coordinate1407.mat 33c4932c538c54da3cbe43f506c5e5ad17c5c3883704fd5630af0ea932d3f546
coordinate1408.mat 0d3307211ba07ddcac855a0b2a5fab07fec1bffa9116c5d3a1725195fa256fc3
coordinate1409.mat 6136c55882176ddcbee8bf40c808df3316bb3ef0572a24844c601ae4dab98190
coordinate1410.mat 617a0bde32933c364510cffc07528952aa8e509c68ea243c049ef73ac3cfc430
coordinate1411.mat 6e41011920d76ba5809dda233aedd09e16ed35f153cc0e48a58ed7a7dfbb6490
coordinate1412.mat 4e8fd20fff8395264116ab117b75334dc281ff24cdee3172364a4a575b6bf47d
coordinate1413.mat f54c36e466464c4981085d9bfa3d9c24b4d9350a6c121f60a33cdafe59d7682a
coordinate1414.mat 47264ba65447a4968f368f26b7bae62c431ae74387318d5dcadf657e34278c49
coordinate1415.mat e8b9a90100363404f7e01378ff1082dcec581f4cf8b77ddd686e0b723e2202ea
coordinate1416.mat 161e98bd826de57d6246fe060a06d9fc96594769d401d535cdd33b7957a5cd42
coordinate1417.mat 96f4b69cd67e683b78eb6614728edae0709445018e8f721b3fda5741572c32fa
coordinate1418.mat 62558301b352f04252b3bb0833922cb12e2c5119d8a1a468008a61b4969b3b63
coordinate1419.mat 42a569048e0022a36afebda080712ebe364385c5d232896de82eff0de6c4a565
coordinate1420.mat 4090f4b886f31012e582bcc6b13252fe0ee43fff7d5b083d07739d6f537b0ca4
coordinate1421.mat 23e6ed1db830b9e1481d73a5166567efc974c6e77a1e6a71e17db04dc0277cef
coordinate1422.mat 1534c05a208f3c2665a4e77b5812a8a72acc69c89c4b687085c11e751788f5c4
coordinate1423.mat 807c4500137c610d757d9867b1f03c0655aafbf27364f827aee3ee69f424c798
coordinate1501.mat 924f0869c7ee67ad4e0b7975e5a7ac554bd361e5e5e822d1ce59ba62b41e95f9
coordinate1502.mat 48517690815a1da10a046ca73140153523df0e4602b037c223b81c1dbe2bf329
coordinate1503.mat bd555aa868ffb19be0d697a89f6c62fb985c2f55aa2057fb1502c4682600f53d
coordinate1504.mat cbec064813c63a73c4df3b5dfe70235c449902f7e64efd9e369993f69d7bff4c
coordinate1505.mat bff1e4efb436d6676053a37315ed35bd603a53bba89b69e1d7b29307f2fb5631
coordinate1506.mat 5db9ab64c92805399f3b11be797c1382d844baaad69f21343e6cb31c6d9fbb4d
coordinate1507.mat 6bd6f2a09e3630c672e1e5e30273b1eec3d6bf3c2c5bab9b966f604e1ac6be18
coordinate1508.mat 86e7bec98ba3da5863037a26b7b035bf1f08de1e5fef3b2380958ca3b0a3db4c
coordinate1509.mat 331d8bd8adb1db6a77b8ee38bda0493cf68307f0d47afce1d5ef2ea3ca1b07ac
coordinate1510.mat 22d637b20e851c16d87e49567fbe2470ff9502e3abba4ac696973cb7be6b91c6
coordinate1511.mat 0d38904f37dea36503daf4150b7cf51cf8a3cf41603e60e4879cc357d2ed8724
coordinate1512.mat 8ca64f0865a185de9f1e5cf09401f61141b16095ca7330adc3572e478fefc00d
coordinate1513.mat dcb1053209ba2fe0ce751ec93d818efe58eb520a272608f92ecb5938c828c83e
coordinate1514.mat 6385a16787e36d793df0b715c2410f7efdf1bbef54753290513387cd2d6dd703
coordinate1515.mat 8691cc5a3a931b5e1c3e8dbddec5b6eb33d0f80c4e808f09139879a8378c4df3
coordinate1516.mat 626e8b7b6a291989a718a832aaddb9264fc6dd6ef4c9494d41b8326704ec5c8b
coordinate1517.mat 1d39e34ea15714c01660935f6a5e08a4046a50a032d18a36a3bef1e532d4e0f2
coordinate1518.mat f88bb2c28d6c89397855bbace2f472ac780451931dc3ac998de26221438943f5
coordinate1519.mat f7b521f5ff57ce10f04b37ef56b752e92ecf04f6a95b78d16ca29b1cdb64d0b0
coordinate1520.mat 4508e232a1fe43278d785d82092c94d39623f5ffd5461bd177c9052bbd317fd9
coordinate1521.mat ee48cd7e6c1e1ab21620a8ec3fa981ebfe41c03c340c141bf3affc776a9e8b24
coordinate1522.mat 813300128841048ad276e9e5dd7f3a43142302c7a1c89c0c9c754ca83d1b3a8c
coordinate1523.mat 43216b537bebcffd064a740217a1bf6844ffb21818b995459158360abe773d2d
coordinate1606.mat 8c769ca8e15b9328f6ca159469f880f10c307973c7d7a3e8ed2a33549db0fd16
coordinate1607.mat ef7019932140449de5ffd0807118ba0e0b9fd3cb6376c7d50ca4504d3eeba078
coordinate1608.mat 9d4344f9bb2da74f32a571cad8dc930624e8b1fdb957d603dd3dedd7b37f8b5a
coordinate1609.mat 5e92835386e5be74178d31d38f8d30da6e00b6c7f7ec3237026daa71fbbc5915
coordinate1610.mat 540a2f53857c0b7722988e45c2b04d5c4be5c44442bd0fd18f74ce6c8915f173
coordinate1622.mat df9a9f48f66a4c624385825265c38f95f3f49c46851a3aef1c33cddf68ad9a47
coordinate1623.mat 5521aff928de5805c8ed11b2ab53dca6fff6ae4ffa64989bb36f26932d733dc5
coordinate1706.mat 46e060bd944904dd8d8b48bf18c28db4d25e8fc7dd62d11a524ac0b1e860d2d6
coordinate1707.mat 4b0107a202eab073b15b912c2cc9e4e7ecd7d666092f6531128cdaf83fe0dbfd
coordinate1708.mat 64c02116f3e55e503bd6c285b9bb29b27bf566ec0860f34774a7e1ca03447190
coordinate1709.mat fc25f65739827361e88c38b0058513f8987191a12d798bbfec327eea65ba1705
coordinate1710.mat 58dc8c518fe7742fcc22b842de5a746c714b9a35e2214961b72eb95b2e7895d0
coordinate1722.mat 39203137c88b7fb2893f302b5f56eb766eecc12031aeda9b1e07d852afc445a6
coordinate1723.mat 8dd6cb149e2e634e53e42e65f2751d0604017b2c2af1f18183dd32570c5b689d
coordinate1806.mat d118fc171f5b3ed709f45c57ed7d6f3d5a5ae3c258a3c781d28ad1bb06e3445b
coordinate1807.mat ae05a7e33cf72f5de277e70ecf657962c7057be8a732a1e18bd145e199b3cd61
coordinate1808.mat 1efed431f07b6f9e3f12fd25c7e480174f729edefef03c122fd07512b68e8be9
coordinate1809.mat fa76b9defac6225e7d89642241341e08bdf5991ed5351cee020accadb3e54d84
coordinate1810.mat 06a5bd47e1658b42048b8082937bbef0e9f2baf5286b53750eea4c68d614df7c
coordinate1822.mat 0c45e0554a66c73833166af7235aaf7f9b161b4a27252deb05559977a00c0cf2
coordinate1823.mat e16847aff172feeb3cf6b02500e56af2fe81cd4b478a056559b5611a41830c0d
coordinate1906.mat a4fa8c299fd75a3e42ee269a849f89422c3b8d65dccceb1ae9150948cb41cd27
coordinate1907.mat 9b524195e12a1c71c691a30d57c217116e16409a37afde16bf805c1cc66645c5
coordinate1908.mat b4ec59744b283785f81be189e3ba4cad9ee0a747204638cccf73c3ac06649cc1
coordinate1909.mat 099ced1d02d614cea10427b83fb568c0ce8fb7fd70dab675a108a2958be65856
coordinate1910.mat 0277dd2689218447714fbda0f7aa1523427c04198be2a92901383daf839a82a5
coordinate1922.mat 7088c54686dc524f038cfed904c71b6c782fbe0c3d921a0be5f2b17d505482cd
coordinate1923.mat 24130cbcd23ecaef1ed9f4a8f74a94fb151a9dfab719a7f5baeb74155e2037a5
coordinate2001.mat b087698763bf1f48ad095000af61c47f4f0e17f0c96beaaa2664021db22b4856
coordinate2002.mat a412402b7640f8335a8d6d1674015699f6f07df815c4e2577087d5733c5883bd
coordinate2003.mat 593d853b049c6e5d16f0a81500230279cb00bf0b629549a6bca2ccbd16ee34d8
coordinate2004.mat 7abb7b77b8ced4b18d3dba411780dc08c8c07eca3be9b13ba3307a5fca4e7683
coordinate2005.mat 4b7d8a5fbf445b0bccaea501e996ef5b72ba55c8332c1c2e55e63739481343e9
coordinate2006.mat ec8c2d038d636d72b2e5a85530bf40ed2960c4a286dbd5d6390cefddf82d9761
coordinate2007.mat 22bef9ee5cb1bac6c61148cc10f482c59f93e0561ef5251b118c6a5081eac3be
coordinate2008.mat 53b6b4fd84ba2b260037dba368fff041c1b6b6b5e1dd80aee473db249b1d63d9
coordinate2009.mat cd497f922c8f380bdb7db49eb448296683f13cd2b39696650fd13fc0c4212198
coordinate2010.mat e855f8c191fa8f24e9f29723808f6e7e9c0a793278eba48ef92a73205c212704
coordinate2011.mat a1202f25033d35078e1f4d926fbdc5edd3631f861b2ca5cf1a33e912628f8af2
coordinate2012.mat 405441f0cbff1ac51e8bcf7ed55524ab27cd8a2999a3c4dbe4e65503520e7691
coordinate2013.mat 63241591bd99def4bf92d3587874de530dd57342d2746d5511985ddd2a7eef02
coordinate2014.mat 76fae44e3d7b70ae2985077cec1f0165eeaa827ed355b99f321fd25513d3b51e
coordinate2015.mat 6e117aff7df7ee89f17aee3e6eef64bc19414fb8e2bb54a2650950774b4cb993
coordinate2016.mat 91583d690c0df64e6084c53c345c6c55669b5fbd208bb8fe3db8102784e4ba92
coordinate2017.mat 5dcea224b8a09dbe0a9564e8ca7db7c3a4603bb0abf66b013c0c9fd503998a6e
coordinate2018.mat 6a154334e539c810fc828e9ce1b0f1c4614f76ab43f268b5711303df805c7ef6
coordinate2019.mat 4eabc84f1df9dfe89adbcae9bc0b154e79efeacb07c59b67122217e978a913e0
coordinate2020.mat b37972c289ef5584e3eafda813fbf731c315ea7b86cd4db194d9a7e18c8c42dc
coordinate2021.mat 6b59e6c42e567b71615582cc13647a03af4521adfd6ee0dcfa7606f7a7e3e01c
coordinate2022.mat da4954fe7641dfd0462e0fc60d94de007ea7b663d1aa088c2d7a36e33513ffca
coordinate2023.mat 832a79dd45d7c94d71ef6575e57e4cf4846a2f5ad79ec53d170c1c65fd191a9d
coordinate2101.mat 7c9ccfa7dbe0ddbae99bbab5309a856918fa80779e6f80414261b721750a2e06
coordinate2102.mat 588dda42c4ce9b4b16ec7087b7cfb9be89f55f78da2fddb01e7c432b1603f1ff
coordinate2103.mat 3943c8ffb3e9a15075eae15e763f3d9c0e23d2c9b9047cfb13ac6369fabf410f
coordinate2104.mat 13e0013c6e7b96e21188148932c64f503c4436cfd8755c6db3884dc28064df81
coordinate2105.mat cead78c20c10899dcabb89c55ab709ed082ec092bd30b80453fa087789e60ae2
coordinate2106.mat c2a59b21c7a223d3fe78d1a2b071e508ee975dbe577f50280dd26da3eb2a9a00
Lab Dataset/coordinate 301-317/
coordinate2107.mat 27d3bfe21afedc59661722d2dcd6c285f912273ae9eb91369014a392367031a0
coordinate2108.mat a493b0aa9122a98a835330d3af32d4cef186f1461f3ba0fbb848080f28c4f389
coordinate2109.mat 81212ec987328b9d1e914c6697418f883e857880a4e27ba20030eadda8fbaab5
coordinate2110.mat 074b7cdc7fea8c34f3fa87cc7f95c2dfd18e48fdcd5f1eb4105117cdf93940c6
coordinate2111.mat a2781374131eabc6fd6563b66233f516767dd5bcd9e96103c7b8eb998511ff29
coordinate2112.mat 265293215713f700e21877a43e456c0948d1f18498c36d9f2b157ab87e0bf2b0
coordinate2113.mat 37a0084f661498c7b44230cfbd797044582312f5eada038fe939209c52651d7b
coordinate2114.mat c8fb85e396b5699c3fe00529832f08e7f53d1059e23d544893145075cec3d59e
coordinate2115.mat 1cd52a8d9651b03431569c3f5f94ecfaca7760821e939a26f2c702751866b2e0
coordinate2116.mat 9838300e73c3c198d6f73fec8f89d00aac2d33650f900933ddf1f3f2b23a525c
coordinate2117.mat 58a8683740df832baedc0dc2fc5dbb3e49e80fe1cd505ef1bfdfa75d7f0a7ed1
coordinate2118.mat 21e6830a9b1b24ebff853355a5b5841bb7ba065974b0de475f03f3301cd89445
coordinate2119.mat 9642afc420717a5a37607f24245a6109ca6a0a5353ba744d81bc96b884a0ea2d
coordinate2120.mat b8b3d6af8e07d52ab613b005696a9a80458b2d8bcd362a184ab86ee242918ca7
coordinate2121.mat 5eb1de60fcd8afb05c292b893f582dcd8eacda7da206912ca34a69dfa92c2219
coordinate2122.mat 2d2cc23b8b0ccf467a33c32302ca64e4f5ead0c9f550960c617b2d5792ef24a8
coordinate2123.mat 1ed3b547b01c68824fd07309aa54d3dc51518c4846a3ca6cd241d91a15bfe7b3
Meeting Room Dataset/coordinate 1-100/
coordinate1001.mat 3134d0d8d969f25322c5f4f803cef30a4935c81d284c115c64187c75dbdfa1ac
coordinate101.mat 54d228304570363ace77cc83e549e5fd96e6f9b7bb82872cb9bdbed18e266159
coordinate102.mat 9b44c34efb684fa88101bd3ce62a10ec6ba3f808bcb05819b1e971bd8a6b67c0
coordinate103.mat 1948e69a89066b182c1fdd8753e0804e9cfecdacfdd9ff6b0b2d57251aa2d3a8
coordinate104.mat 20bbf0f3e54398c8dc61fdfb9d31e84889f251e195f195447514c3c09fe08f67
coordinate105.mat fe6ef99fe8e2bf4defac87003e33bcb8d3888f177c90850daeb3b92b319a94db
coordinate106.mat 693d7e38d82f55d4f674d29f951bca4c2b3aee49538d2b77eb416e6c112f42d0
coordinate107.mat a7ae895dab76e2066cafc204cbeaf1462d20cadd115778389820f23440570a5a
coordinate108.mat 2aef70920e9d4225ef5700beb66ba90ea83e527a23ff51d8286bd0c20d8442b0
coordinate109.mat 4852e3fa462054b4b2c423e2d43ae1f414642eb9c599d489e8a8cc5ff6e46152
coordinate110.mat 6e23e83f67b82945924b408c0d03ab4b5d773778d4dd9afd14b6b6b79bd80b55
coordinate111.mat fda68d0bee7d27183e75d82ec8580844012af88fc8893483d374f7e77c04b0b1
coordinate201.mat fb0e19d2ce2fe54f9889e9ecab3d41c1f00255ab73495a6e7fba0fd2ab91c410
coordinate202.mat 28df6bd04928cc328ade4f9d3b4f8e7da3abedb1c9ef78d1580493883a49a7b7
coordinate203.mat b97213ff973e69f7a36c27cf172e681ba9eb31b36041b214436d8cf6989fe2f2
coordinate204.mat 0521bb8befe2e1051ca47cdbd7129537d84c9fba539d5a1bfa29444828e976a3
coordinate205.mat e2e0fe649dd2455696e93388a56f52a55554114ce971a68c0fdf753b7fd1cf7c
coordinate206.mat 01f810c3d490d5898d055b4c117d27bbc87ad1607143706811500d8ec7832eb5
coordinate207.mat 093e3925a91508f5fe5149ec51ae00c87060038d8df4174b7a08d30f1bf0500f
coordinate208.mat 74603d1f372b94288d0607cf18f51f5ba5db362a3c213da44d41e844e7b35db0
coordinate209.mat 876b493d72c45222e1dbed8623cb2a6d5b6d9059fb60f1cb17fb92d5f16f4ebd
coordinate210.mat df757d6821d401aa5417685eb50e4363d71e4281aed13b98d0c1b8283342aa75
coordinate211.mat b45278b380c890f691ec889cf2e2916ff0ac1d18c19442d6178afd1ace92f7b1
coordinate301.mat cf66a9c5aac79b785a1d0cf2a3e1697d49d7510655e3c737470393073e84c02d
coordinate302.mat 878a630ecf5164982598a5b3fa9a83f4b4cd9ae8738331f5cf89ada617269eb4
coordinate303.mat 188a5bc09a96f0afbb8095753cb6d97be4aad90434395654f9e016b8905a07ee
coordinate304.mat d8f2ebd19471a0435dec83886661b79869785ba91676487b55fb6e7699f633c5
coordinate305.mat ab79109832ebd43510a89510d8df5f676a2c5706c4ee81043f4ead65a838b540
coordinate306.mat 19312b3d6e7a4f1f02b78506afd4f36a78563ab744d6ee6483989e34ea942e14
coordinate307.mat 3ff183edb0e2f36b8006f656a8e45aaf1ce2f4484f1d28a217ded5bdff528d19
coordinate308.mat b0b6a106e737acfe7be678e6825101d993bfc5cc5fbfd195ab86685745362273
coordinate309.mat 62a80c7caf6cb653d011378915ecf3a05ed9dd66d70bf3850200115b4fb10101
coordinate310.mat 62bd3fdbdece59edbf7a6087fe6bdb958987fbe2c767ba8b2bc08cb141e4e0d8
coordinate311.mat 1c1f9340fa6c07b84adc7ac609e87a7120cc3d12c92a478569f9fb16b54d92bb
coordinate401.mat 5c365a92e08dba5be4bf4995f41fa6167e9c17a96c952fd406a32b00d1e15794
coordinate402.mat 0800c65c017064c373977d9d31dfa13e80263c12f3e2b9bf0e9a3c6ad2c00141
coordinate403.mat 7b6e8a8d13a27f3e7f35f752f0ed9dffd12c760febbf23a922c29e44d3a828c0
coordinate404.mat 72630f4a0dbb8149a4cecf951292a6d36643ce3508b38acc2e0f8bc20031c584
coordinate405.mat bdc143b254c84d507f72393641206515641fe2baf51a2e72d81e38c4626ef050
coordinate406.mat 62d2409eaf9f20828b2c047109cb22487f05d9ae3c0574a89b5fcee2aa215253
coordinate407.mat 260c6b8d351c7926fd74e224f0369ba6310312914a4a7eb414fd507ac06deeae
coordinate408.mat 2908ef9c00f72b23745b94d26d802b650af48397d94f6ec5d52c81406ae39c54
coordinate409.mat 8ad0208190cdb40ac0fbcc2ad825c21e455f600f221fad601652b4b629645688
coordinate410.mat adf4b222a0ffecbe7a4aeedaa9fc545af905bad9c0c4852a202a03437b0c81ff
coordinate411.mat d95535ff65564f632ca0adec44bf1611324c0ecb1900c50a244f7a58d2f35267
coordinate501.mat 199cdb2e9eb1e6c60e526035e100d8c7bfb12d024e9c000f5fd0a697e5d18141
coordinate502.mat 1d74df28a2567b6c734a6653507f7b2580d743bf01766fb4ffb98649a636c42e
coordinate503.mat e11d2106f5929af1a2ba4e705bae1793242118c73450776ffaf51cba6efcfc75
coordinate504.mat f0d57f147eccb4c2ca205a2c62c6547e99c46700788cd8e86d43ed80163bc5da
coordinate505.mat 3e8eeaad5dfdeee701ca900e0a653e58ee7b1c305dc7ad3d54c4f7736400de6c
coordinate506.mat 00c04e7a21d3031c3ba6328af8705a11b23bbe1de01e4fec5b50869e3b2ef9bd
coordinate507.mat 4199f6419e0588f909276af6072eedd442b7b2020b66c329170b6eaf9295181c
coordinate508.mat 2f588dd489e994f418133b3bb521911c571f0ae454ce6eeff2854b81e886459b
coordinate509.mat 55efa6e046aec6e94211ee87ea3902d272176687eaad33d076c437871c09b6f1
coordinate510.mat 40773cc267293e4836c6e270d95b8420af0d76be16d4d1c204e572389856f934
coordinate511.mat e09af9c4d3a2d3f48178a072c924f0c12d5d6498ba45c710ac336f0ce31b947c
coordinate601.mat 0238c6f2a7330fa12de6c7a5f283d7f1801a15a91140ef8ce19c35c6416081f8
coordinate602.mat 3d1aa85fc0b66efe16be32efde94bdf19ffdc8a4ff32db38287391453155dde5
coordinate603.mat 41c7e8b9ef814a13ce885b714ba6a3d6c265c8d544ca4f7daee4c9db5d80c078
coordinate604.mat 3a046d9f93415b9822d8056a3a7a6c2dc113e8a23a905a77fccfe188ecf99760
coordinate605.mat e37f2410a87a626ac6c828b155a551924909b740c715e6559d363c2874a4afca
coordinate606.mat 1b1f959a837e0ac6bb03e305d2d972a092ab0190781c55d9755da3bca4c5d94d
coordinate607.mat 62679f51d8ef25b57e428d6b26d7f6091bd598ae154fa9b434bc39deda32e6d1
coordinate608.mat a8ef7b9b0e78ff872a56409bf1ff99c90553721d2231bd5075e364c06c142974
coordinate609.mat bf026d58741a968f8a75b330925e798762896b10fb9ea170bd46f4664c248d6e
coordinate610.mat df97f31589b26d62c46199451bf4ca9810106b72a36b519f8a4b26ca20a25c17
coordinate611.mat b9f181592064f6a5314a0f66de023ab007639c0c14d9fe594a1499ab5dee6395
coordinate701.mat da650dbf2d77d61fbe8e33ba32d9c266910cdff31a31a9155623080502988590
coordinate702.mat 2d96f23e328754cd803bb17317663effebe0930a7387c1bb4081fa8b60d203ea
coordinate703.mat b641ea21d425e081b8df087f638a1eeb8cc7699016da4472e1e664667e9df1bc
coordinate704.mat ea03f9907d11f4d2b34c499a4acb4aa4d8a8380e36862109ca37a1bbcd225a85
coordinate705.mat f8fbaf11e8afb17a10ffcfd95e7e189dff604b3bd5ee7bbcfd2967eba7bb7d38
coordinate706.mat 40cd8d6b262d01a565da103db77383b3ab46c0b90550c6a69dbc0f79534ebe69
coordinate707.mat 13eedf3363e60f24b5d85b770525754cc9efa4415ec58349323dae57e06bce38
coordinate708.mat bcd806c9ccc084bb465a0be7032b33eeb77a9c1935f8cb4d0eb322e25a0532af
coordinate709.mat a90dbec191265bf236b29028acca9c158dbdbaecff3aec0db25c6fb9488d19e0
coordinate710.mat 32d8551f36bcb790a5b21426d54ce15f42c9cd7e3affa444f3ecc390641493d9
coordinate711.mat a96e92a270d04d6760497b2db9f3ce6477ff4ffcfa77be34b8544b3d69d74910
coordinate801.mat 38d78b4b589a43ec1ac5290c8e792acf1ea3d6a4d0650b66d87105d60ab072d0
coordinate802.mat bb79e8e00caf442cd8b49466b34a05ede095989f4c66d7d4f4fae24375a6f3fd
coordinate803.mat 80fb024717a0f3bb4b6c180429cc89453d6dd56257ab892220fb6890ea37574e
coordinate804.mat 134d8041b60f4dc890d5a93969d750b372703def673cdce994a5d853b3d310ad
coordinate805.mat 7f6d57754fd58cb80bc415981d5cb4a9577962697a520e25504f7d81dd98f965
coordinate806.mat 5f951835e70e44bda569878716ece06d5860aa99ec9fc647145ddb8266cce25f
coordinate807.mat b5320f1eafc3a009d364beff4d23f2ef97df6b892f6697c144bea78c4f5d09ae
coordinate808.mat eb53fae1d6c910171066895b476ff556ae47dd28deea68b08ae084d231e01cc3
coordinate809.mat 726abb1dcffcf03d4204d381799b6aeaf50667759b1d4e202c4a103dc478d4b8
coordinate810.mat 23a59227476de571a7009283c0419b6f68beba0c8424a2c6cdafc37f9cc8a1da
coordinate811.mat 1be0e5fcde5c185fd4feabaa141a45de4c389c7698453db5fb1957da0e6ca88e
coordinate901.mat 66654b84d02b5a8d40e86028780132b4eb4fae4a85eb67218ef2e7f922a216b1
coordinate902.mat 3d217e8d5fe665a1982f2113c3c967a0ca55de16027397519e476a86546df968
coordinate903.mat f2ec8e3585a548ad18f9a2f963170b8441339f59223b937ef7694ef238ce668e
coordinate904.mat 1f01043513f77157995108f8b0d3e079a23c66bce79603879d2e6d9f1a61083b
coordinate905.mat 9f4b440612b341ca10ace3f18cb7ed31943151ba7fdc29c53275e172ebf5de47
coordinate906.mat b11b6071044b8bd0ac3206aa4c50ca1fd84c0f5b74ac69bbc762b52bd1d52847
coordinate907.mat 34a6e9702572815102a761c42fe17557f95737c753ce62bcce22e0c784b33425
coordinate908.mat 7bf7a8b94fab4ac146482d7e1cff30fdc5ccefadca9b29bc47cbbb31f2cd7c21
coordinate909.mat 933fdca473c4c654ae0b5f4c339ba6398a73216871fbe58d5acbe95f96cc81cc
coordinate910.mat 292bfa5eb4066b2e6b743dbd42acc6f3fa92e265fb2a0802dbf84d28b12d61d2
coordinate911.mat 2dcff00331299edad2aae6cecae3fb4f591a60add3bad1ef45fa4b7b0ef0f9ab
Meeting Room Dataset/coordinate 101-176/
coordinate1002.mat 1f6452e0e98373d901f4b919a43aeaabd9eccb477b215fd670fef5165047cab8
coordinate1003.mat a585fe3e9eed9e53aaf2b9361c14281860fb69024751dfe65150e42023483a8e
coordinate1004.mat 2a58ed1f409357f67cc09d633fdb0e13411be40a0009264c41f4d5794dba7034
coordinate1005.mat 3e08fca3c4ad7634e85bdbeb0f13b94ebf5128bfa4e6eecf900a07b39ccd35de
coordinate1006.mat cff73feb68ff938b43829c8be8d54d8c4a3f76bb07d1d11e6ed6e4e73093688e
coordinate1007.mat 2e72618d6976ccfdc632cd762259c6eb9d217f96695b76ae57a51dc754375bb3
coordinate1008.mat a4ad60dd95e681bba36722c95ab5acfe0500d25bebd1f9856ab95334fdede209
coordinate1009.mat 1777717c19db3be454d1e09b312deb04549e8a7c66631b79455c94ce17c141bf
coordinate1010.mat 82feb8257ba0e42314b5f7f0436cc870377019634f11f51a04bd37453840f4e6
coordinate1011.mat 5b30c45b3027563ef951d157a2697cefcc4830e166f4c50cf6de60b410732b50
coordinate1101.mat b820dc6aa4d44545df2a28db096540e30307290140e93144fc4b528996273c92
coordinate1102.mat 98e1b77ddea8beef419247ddcdb26345b4694c671f73c1d138891e6b6fcdd4c6
coordinate1103.mat 5e745efb982c9540a41403267004c7bb79c20b631ee91262c26fff65fa41a5df
coordinate1104.mat 4b26120ef1f5d3d9b160d225b8b654bcfe47b14d2934389b6cf1609695decc7e
coordinate1105.mat 08b95eedf135a38924ddd2525ca404b347ed6abb799d6ecf5477aa9dc2e2f9f9
coordinate1106.mat 35ec4abd5a08d1c5db4714368e0644f699a51759e277a2a8f04a222a7513340f
coordinate1107.mat 2f81b5631255d65691bf8e16712f39d809757d141047f7443cc77bfd12a070d0
coordinate1108.mat 590b370fcf7d642dd2ce870f7944a28f4828d21db14c08e5e70ea8260ff87bd2
coordinate1109.mat b46d1f2add644468ad3b80c0efa430f907aed49cccf6bd41cc9a632439e93daa
coordinate1110.mat 6b43ce27c78eeeed23ea13651f44cf0a1935f20cc5bca4800e43fdae12f7ef20
coordinate1111.mat 2e8e81700f292d997b1f2cdd9eacb9148e93b2a9e18f8d921f5777f6137e9db1
coordinate1201.mat b86def8fd5ecf9b53e94bb92b887aa527e1fa6d1456e86128aef4983dbe0e8a5
coordinate1202.mat cdd15a532a736b0f0af219d69df06deb3fa8eae8e9bfd4aa52665600a086a7b3
coordinate1203.mat 0ef15855961a75eee44aed07be7a806c6b474e9aaaccbc1e48e2e8fb3060b574
coordinate1204.mat 9aebd6b1fc8ca6f25bbf7c9a2b5251bcaf6672d14eb417ae426ff1128152b464
coordinate1205.mat 0e0ae924f17ceb3de1b158900785b9b3fdb08164fd95dc3f18185b21b3ad2893
coordinate1206.mat 11ffb6d11d09c70965088f0de88f39529b8a9da439d5090d189a01f0e58ed6b0
coordinate1207.mat c78a9353d2c9358e0068fdfe7ed313b5827b89241d91aac0c6ed4127bca5d12f
coordinate1208.mat cae8ab915b63f9ab02a1f7ebfccf35d3a992ef865c2186c5e968a31117ca373c
coordinate1209.mat 1644512673f820898b61796a001075495655f1ee3a1a498394fb1c007d0716e2
coordinate1210.mat 444521374177c9f231238e8e082c081292c7a41c34e4e5b8f194a56715580629
coordinate1211.mat 8799001448cc194b9370120143d2e68b774cbfcdfab73e5f9fdee46fb1d34b7c
coordinate1301.mat e7fa9d4d8db9fbe65e607755347f700ac38a6d4213af51e726f4e3f90decdce4
coordinate1302.mat e2b238485786064279895fc5745b094d0faec6d5518c3ceb5eff0762108cc3c1
coordinate1303.mat f027e6334778b78ea56875197294f6f4c08cc546f004ca625ef767cd3008a568
coordinate1304.mat 771dc6fccae23615a6d6b7ac631490ada107437f74266b579d7bce09127062f8
coordinate1305.mat bca847152e085d8676d79166a8969b5c07b6bee2c0b1d7c57bfe3960255fe3fb
coordinate1306.mat 25b78a1e7ea5bbc4207c33a4c64a82c6c5151e3a4a0069c5b9b21b0e9102d73d
coordinate1307.mat 299bfbc2ea0f76390fdebb1cefe7c4402cda6fc1e274b7f4876bf9d2f2396429
coordinate1308.mat 3f1c48972d447fa3c4866cfa27520eb3790af2d607a752ea6807c8580942b225
coordinate1309.mat 814c61b771241824cfa9d1d8130c1fd045c6b8abd6d9945dd6d42ab9bf9c73ee
coordinate1310.mat 1420e7496676d141bd6b63815718982351f573e99be0090e0d4c8cd46530db85
coordinate1311.mat 1dcf5ecb0b2f28072c02218716af55e8f7b87fa09d4730dd0d56e6277314992e
coordinate1401.mat 12eff44ebc0c174c75c2be6aab653dfc769aeea6e349d6e4b5257dde15ddeb3c
coordinate1402.mat 9ea5e033057f6dc8a5914eb10047bd1d354fd79ac20fae6328d447195371af27
coordinate1403.mat 8dd28219d8b7adb3f23baaa409bd1d5bc9bcbad61fc37dc619331bae38946ef5
coordinate1404.mat f14638a167f41df49c188a11010003d6e0fc31948f1332394b23198be46568d8
coordinate1405.mat 9b2c3da6df927afb093181a0f37717cc4bef55cabfe60b5480342116c73eecde
coordinate1406.mat 069fec0ec1f1f2a07f80a01c1135e63077c3cf1b4e621ff859bacf400e029dd4
coordinate1407.mat 717412ebc85dabd5b583a5da8b000c9867190e96312e7468c34797e6d67647c0
coordinate1408.mat 6384a958a2ca2d212d3891813cb05b37e171f4629a9e6f84322c47ee86d99125
coordinate1409.mat 9d6b58cbcbf93b72329c9b17c5baa3cc6c63aad550af47287035728a84fd4f43
coordinate1410.mat 5af4bfeb716b2955fc177aba23fd460542ab46afa6adefac538718c2084b953d
coordinate1411.mat 69c04720e22968cf7cad29dcf6fac0963508c8f891433c405c95205f1cc99844
coordinate1501.mat a7a8c50c05a3c84b19413645622e554e3764cb0780dab117e4e7473952e660c3
coordinate1502.mat 536a8aae6d10251721e2323c345838507a28b8c0315daa0eaecb1531164c1efc
coordinate1503.mat 4b37a422990271cf09ca230775f80e5bab6195e17aee9057ca8586d699c7359b
coordinate1504.mat 4941a4d1f78956bdc89e3c3d9b7cbed6251341f46f1b59c9e47c9813e4fba3fc
coordinate1505.mat f7a2f015f01e8a50bcf0d35793115fa584db5fe6ea8272900dd16eef2d722405
coordinate1506.mat 97ad85e1ddcec4e0d6e47c66f0f1607a113ff35aae5866a559ab406a5331f7aa
coordinate1507.mat 51a34df28c834e79fd9aed4b22140a866dff882d1738cb194bc3cd0cdbc8de77
coordinate1508.mat 628093712d4d2e59a686ab562f049336b3df0cbe64a044a4220bd09036c6ccfe
coordinate1509.mat 378d9c07d8a6ec95582b296e233f8ce6789d54fd42d237ce9161f4f699dac879
coordinate1510.mat fe09951501ac042b93d72dab42d9384f96a97519336f0c6bd86c4c2a84b4dfdd
coordinate1511.mat 33ccc9d3c710a821e871ec955ea6611eaf17b41ccf6975837047726979280aa9
coordinate1601.mat 0b1b011c0cdf195a78740654fe9f28189b973afa3e7dd5aa084430b2593baad3
coordinate1602.mat ee71a9e7d9cee5454afec11acd16263b363c3dffa800f102c36f258dd5831df8
coordinate1603.mat 8e3b5b08a5e1b1e6678fc2818dbd6ed827a9319f18273b59b588173a3820824d
coordinate1604.mat df8a16572468db8184f2213c3bb02d9d748a4ca9b7999af01cce38b8c26e8c96
coordinate1605.mat 3e5d24301f0ad5b0477c76b6ac6e6179afb051289a4d8d072dd260187cac962e
coordinate1606.mat 5d7db618681ff35ea9fff889106c55ac5402b2b48fa754218ee8ef1d0624b82d
coordinate1607.mat 58709231f55c4e330c85b66e70f54418ad7736f0d572340896ffe3e70773dfa4
coordinate1608.mat 1b5c7a1d628027040813942bbcabb6f956e0c35b88bc9f9eb6918461ec201c1c
coordinate1609.mat 7e7f81a4406d16c31cd5e9b28f824701f7b9bd21f9ee4970c5fa53f961b91d09
coordinate1610.mat f330475bae3ef1f6cedd7c3c4adcc0806fc9b0cdca983c4cf32f21ec436cb84c
coordinate1611.mat 380cf40acf6bcb73a5552a463e69a3b9c35de5c9d923f9e020e549cf659d8a68
miniLab/coordinate 1-35/
101.mat b38796e128e9b5a67cbdd95e2b82dcb7f92db218055db9c6431d84759f506635
102.mat 981815563968bd2f9fd45fa900ceac003f346290cb1eda0fdf07683fb5bf1e61
103.mat 58d3256637262d766eed06bab4fa805f3350df555e94cd76be966df835d727c6
104.mat e0417965f483423decfcf83e12af1de69051f8d739781afc27382a616eaf6e0f
105.mat 2973808a7e93060181fca5531a33a1e62b4995d4a16e16bd7cc3841b4ba796d4
106.mat c696b73fa6e5149fa5e895e5b9b4e41c233fefd049f0df1d9a06e6cea98564a3
107.mat 0999c7e1acb30cbbb313aa9605062cce4848de9237b0eac765b847584998bc29
201.mat e1d06990954ff733d3599420743cb4f73edb5e5e8c9099b660b6fcf42aa0c06a
202.mat e5c554ad50ce421b07509982ab0fad7b3e9b14be9bd93de458142a539e024cf7
203.mat a434403d7737391ebc05d398a2c6c58ae27132f7ee7b8ae044abda59085b5d2c
204.mat 95583c2f97dd6a62a82674ff8c83bb6a144f14371a69969c04488d8f61b90302
205.mat 10b2227a50cc873508a12b950237d421abb9d35fa5e01454997ce85ea89267a0
206.mat ed70b910daba8136ec8b7a3f18f4ae4c1fe5e9d9afa94933de2daccc939a3c81
207.mat 7ff041f7bcc432481a879c980835c0ccc70997522197286755f1af189bb45c3c
301.mat 0e49ea6490f873e371e4648d6c589f1e601ce9b9c6b9220925dfcb59ad53530b
302.mat c4085e727b1135564471f33d23d16fde1c0fc1c8b85762cb4ceb79e8e73500b5
303.mat bc7c8aa223fae986bf9c7d298df6995010908932569ef84af7d0892898e0e4d4
304.mat 5adc3e08b1b4150c19ffb3f79c7bff9ae5c2aea31fa2f076cd2587a8a933f481
305.mat fef9532438c653331ca631a4a0a469f0afa72a198fb8311aeeb25107524081b8
306.mat c99c84300b58766f2736b8ef416472b3493eed63a4988b8348bb865a8d8ace74
307.mat b91c7af6dd6a5899ffc82a786722fd7196e64a864702855992d4b260060c4755
401.mat ccab76eb1eaab10045d974fead3debdf448d37d29129e1c19221c3c683129fcc
402.mat 7d5902ae0b9a7ee0fe8866fd0e0d5650a3afff490ffa3597cf9f5a063893c4a5
403.mat d6c13984855af88ac798327d1881283f6b67910f3f117c8ce859865f4e6362e3
404.mat ade9cf8da04c804c0cd154f189ac04d6d6bd4a8933689ea20c18af1119fff1f9
405.mat 98b021efba166f37387bb44c3a6351345920de971144f7899143d263b5f0190b
406.mat 97b0c0a1940c11d19c0c9e8c3a846da49ae64521c08864a744c1a2599c80fb37
407.mat 6b6da75bca4d9cd02241fdd54f98a74ecb6d3654ec4d8fec17877180f8fefb87
501.mat 37ff02dfc7397d2b068af3095701d7e438f154c5fcc38b6fa6b735d77827bdaa
502.mat f70995c880d0cf9f0d4c0e5d936d62b5d9ee0a4af729d74f3423f95fae75d01b
503.mat 80501e652142596dd00c43ebbf8ccac4d12efaa246007acb29cfae2011e92bbd
504.mat 52f1921aaa628945e9ceb0b23c7bf6f89e00051f8179006b27c8c40709e1cdea
505.mat 42f0219326a585b910f11a51f18abf37fbdc0752e4b2f409db2d999f5eabb8a9
506.mat 1d7ed820cb19f892bbcd5345976f40611d8dfe191d7e67e76e4c6ae7464da761
507.mat 59b45553aa5afdfc2ab6e6ea6ae8b3f6c74ddba120dc8683ae1e9ab543248779
"""
CSIFingerprint.files, CSIFingerprint.urls = _layout(_MANIFEST)
