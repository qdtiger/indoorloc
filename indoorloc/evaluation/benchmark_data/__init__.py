"""
Benchmark Data for Indoor Localization Datasets

This module contains published benchmark results from academic papers
for various indoor localization datasets.

Data is manually curated from peer-reviewed publications.
"""
from ..benchmarks import DatasetBenchmarks

# Import all dataset benchmarks
from .ujindoorloc import BENCHMARKS as UJINDOORLOC_BENCHMARKS
from .tampere import BENCHMARKS as TAMPERE_BENCHMARKS
from .sodindoorloc import BENCHMARKS as SODINDOORLOC_BENCHMARKS
from .tuji1 import BENCHMARKS as TUJI1_BENCHMARKS
from .longtermwifi import BENCHMARKS as LONGTERMWIFI_BENCHMARKS
from .ble_rssi_uci import BENCHMARKS as BLE_RSSI_UCI_BENCHMARKS
from .wlanrssi import BENCHMARKS as WLANRSSI_BENCHMARKS
from .ibeaconrssi import BENCHMARKS as IBEACONRSSI_BENCHMARKS
from .bleindoor import BENCHMARKS as BLEINDOOR_BENCHMARKS
from .csifingerprint import BENCHMARKS as CSIFINGERPRINT_BENCHMARKS
from .csiindoor import BENCHMARKS as CSIINDOOR_BENCHMARKS
from .magneticindoor import BENCHMARKS as MAGNETICINDOOR_BENCHMARKS
from .csi2taoa import BENCHMARKS as CSI2TAOA_BENCHMARKS
from .wildv2 import BENCHMARKS as WILDV2_BENCHMARKS
from .wificsid2d import BENCHMARKS as WIFICSID2D_BENCHMARKS

# Registry of all available benchmarks
BENCHMARKS_REGISTRY = {
    'ujindoorloc': UJINDOORLOC_BENCHMARKS,
    'uji': UJINDOORLOC_BENCHMARKS,  # Alias
    'tampere': TAMPERE_BENCHMARKS,
    'sodindoorloc': SODINDOORLOC_BENCHMARKS,
    'sod': SODINDOORLOC_BENCHMARKS,  # Alias
    'tuji1': TUJI1_BENCHMARKS,
    'longtermwifi': LONGTERMWIFI_BENCHMARKS,
    'ble_rssi_uci': BLE_RSSI_UCI_BENCHMARKS,
    'wlanrssi': WLANRSSI_BENCHMARKS,
    'ibeaconrssi': IBEACONRSSI_BENCHMARKS,
    'ibeacon': IBEACONRSSI_BENCHMARKS,  # Alias
    'bleindoor': BLEINDOOR_BENCHMARKS,
    'bbil': BLEINDOOR_BENCHMARKS,  # Alias
    'csifingerprint': CSIFINGERPRINT_BENCHMARKS,
    'csifp': CSIFINGERPRINT_BENCHMARKS,  # Alias
    'csiindoor': CSIINDOOR_BENCHMARKS,
    'csi': CSIINDOOR_BENCHMARKS,  # Alias
    'magneticindoor': MAGNETICINDOOR_BENCHMARKS,
    'magnetic': MAGNETICINDOOR_BENCHMARKS,  # Alias
    'csi2taoa': CSI2TAOA_BENCHMARKS,
    'wildv2': WILDV2_BENCHMARKS,
    'wild_v2': WILDV2_BENCHMARKS,  # Alias
    'wificsid2d': WIFICSID2D_BENCHMARKS,
    'wifi_csi_d2d': WIFICSID2D_BENCHMARKS,  # Alias
}

__all__ = [
    'BENCHMARKS_REGISTRY',
    'UJINDOORLOC_BENCHMARKS',
    'TAMPERE_BENCHMARKS',
    'SODINDOORLOC_BENCHMARKS',
    'TUJI1_BENCHMARKS',
    'LONGTERMWIFI_BENCHMARKS',
    'BLE_RSSI_UCI_BENCHMARKS',
    'WLANRSSI_BENCHMARKS',
    'IBEACONRSSI_BENCHMARKS',
    'BLEINDOOR_BENCHMARKS',
    'CSIFINGERPRINT_BENCHMARKS',
    'CSIINDOOR_BENCHMARKS',
    'MAGNETICINDOOR_BENCHMARKS',
    'CSI2TAOA_BENCHMARKS',
    'WILDV2_BENCHMARKS',
    'WIFICSID2D_BENCHMARKS',
]
