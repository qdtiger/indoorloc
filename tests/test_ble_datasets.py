"""Tests for BLE datasets (Phase 2)."""
import pytest


class TestiBeaconRSSIDataset:
    """Tests for iBeacon RSSI dataset."""

    def test_dataset_class_import(self):
        """Test importing iBeaconRSSI dataset class."""
        from indoorloc.datasets import iBeaconRSSIDataset, iBeaconRSSI

        assert iBeaconRSSIDataset is not None
        assert iBeaconRSSI is not None
        assert iBeaconRSSI == iBeaconRSSIDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import iBeaconRSSIDataset

        assert hasattr(iBeaconRSSIDataset, 'ZENODO_RECORD_ID')
        assert hasattr(iBeaconRSSIDataset, 'NOT_DETECTED_VALUE')
        assert iBeaconRSSIDataset.ZENODO_RECORD_ID == '1066044'


class TestBLEIndoorDataset:
    """Tests for BLE Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing BLEIndoor dataset class."""
        from indoorloc.datasets import BLEIndoorDataset, BLEIndoor

        assert BLEIndoorDataset is not None
        assert BLEIndoor is not None
        assert BLEIndoor == BLEIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import BLEIndoorDataset

        assert hasattr(BLEIndoorDataset, 'BASE_URL')
        assert hasattr(BLEIndoorDataset, 'AVAILABLE_FLOORS')
        assert len(BLEIndoorDataset.AVAILABLE_FLOORS) > 0


class TestBLERSSIUCIDataset:
    """Tests for BLE RSSI UCI dataset."""

    def test_dataset_class_import(self):
        """Test importing BLERSSIU_UCI dataset class."""
        from indoorloc.datasets import BLERSSIUCIDataset, BLERSSIU_UCI

        assert BLERSSIUCIDataset is not None
        assert BLERSSIU_UCI is not None
        assert BLERSSIU_UCI == BLERSSIUCIDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import BLERSSIUCIDataset

        assert hasattr(BLERSSIUCIDataset, 'UCI_DATASET_NAME')
        assert hasattr(BLERSSIUCIDataset, 'NUM_BEACONS')
        assert BLERSSIUCIDataset.NUM_BEACONS == 13


class TestBLEDatasetRegistry:
    """Tests for BLE dataset registry integration."""

    def test_all_ble_datasets_registered(self):
        """Test that all BLE datasets are registered."""
        from indoorloc.registry import DATASETS

        assert 'iBeaconRSSIDataset' in DATASETS._module_dict
        assert 'BLEIndoorDataset' in DATASETS._module_dict
        assert 'BLERSSIUCIDataset' in DATASETS._module_dict

    def test_ble_dataset_retrieval_from_registry(self):
        """Test retrieving BLE datasets from registry."""
        from indoorloc.registry import DATASETS

        iBeaconRSSI = DATASETS.get('iBeaconRSSIDataset')
        BLEIndoor = DATASETS.get('BLEIndoorDataset')
        BLERSSIU_UCI = DATASETS.get('BLERSSIUCIDataset')

        assert iBeaconRSSI is not None
        assert BLEIndoor is not None
        assert BLERSSIU_UCI is not None

    def test_ble_datasets_inherit_from_ble_dataset(self):
        """Test that BLE datasets inherit from BLEDataset base class."""
        from indoorloc.datasets import (
            iBeaconRSSIDataset,
            BLEIndoorDataset,
            BLERSSIUCIDataset,
            BLEDataset
        )

        assert issubclass(iBeaconRSSIDataset, BLEDataset)
        assert issubclass(BLEIndoorDataset, BLEDataset)
        assert issubclass(BLERSSIUCIDataset, BLEDataset)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
