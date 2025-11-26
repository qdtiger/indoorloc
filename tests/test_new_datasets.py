"""Tests for newly added datasets (Phase 1)."""
import pytest
import numpy as np
from pathlib import Path


class TestLongTermWiFiDataset:
    """Tests for Long-Term WiFi dataset."""

    def test_dataset_class_import(self):
        """Test importing LongTermWiFi dataset class."""
        from indoorloc.datasets import LongTermWiFiDataset, LongTermWiFi

        # Test both names work
        assert LongTermWiFiDataset is not None
        assert LongTermWiFi is not None
        assert LongTermWiFi == LongTermWiFiDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import LongTermWiFiDataset

        # Test constants exist
        assert hasattr(LongTermWiFiDataset, 'ZENODO_RECORD_ID')
        assert hasattr(LongTermWiFiDataset, 'NOT_DETECTED_VALUE')
        assert LongTermWiFiDataset.ZENODO_RECORD_ID == '889798'


class TestTampereDataset:
    """Tests for Tampere dataset."""

    def test_dataset_class_import(self):
        """Test importing Tampere dataset class."""
        from indoorloc.datasets import TampereDataset, Tampere

        assert TampereDataset is not None
        assert Tampere is not None
        assert Tampere == TampereDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import TampereDataset

        assert hasattr(TampereDataset, 'ZENODO_RECORD_ID')
        assert hasattr(TampereDataset, 'NOT_DETECTED_VALUE')
        assert TampereDataset.ZENODO_RECORD_ID == '1066041'


class TestWLANRSSIDataset:
    """Tests for WLAN RSSI dataset."""

    def test_dataset_class_import(self):
        """Test importing WLANRSSI dataset class."""
        from indoorloc.datasets import WLANRSSIDataset, WLANRSSI

        assert WLANRSSIDataset is not None
        assert WLANRSSI is not None
        assert WLANRSSI == WLANRSSIDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import WLANRSSIDataset

        assert hasattr(WLANRSSIDataset, 'UCI_DATASET_NAME')
        assert hasattr(WLANRSSIDataset, 'NUM_WAPS')
        assert WLANRSSIDataset.NUM_WAPS == 7


class TestTUJI1Dataset:
    """Tests for TUJI1 dataset."""

    def test_dataset_class_import(self):
        """Test importing TUJI1 dataset class."""
        from indoorloc.datasets import TUJI1Dataset, TUJI1

        assert TUJI1Dataset is not None
        assert TUJI1 is not None
        assert TUJI1 == TUJI1Dataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import TUJI1Dataset

        assert hasattr(TUJI1Dataset, 'BASE_URL')
        assert hasattr(TUJI1Dataset, 'FILE_MAPPING')
        assert 'train' in TUJI1Dataset.FILE_MAPPING
        assert 'test' in TUJI1Dataset.FILE_MAPPING


class TestDatasetRegistry:
    """Tests for dataset registry integration."""

    def test_all_datasets_registered(self):
        """Test that all new datasets are registered."""
        from indoorloc.registry import DATASETS

        # Check new datasets are in registry
        assert 'LongTermWiFiDataset' in DATASETS._module_dict
        assert 'TampereDataset' in DATASETS._module_dict
        assert 'WLANRSSIDataset' in DATASETS._module_dict
        assert 'TUJI1Dataset' in DATASETS._module_dict

    def test_dataset_retrieval_from_registry(self):
        """Test retrieving datasets from registry."""
        from indoorloc.registry import DATASETS

        # Get dataset classes from registry
        LongTermWiFi = DATASETS.get('LongTermWiFiDataset')
        Tampere = DATASETS.get('TampereDataset')
        WLANRSSI = DATASETS.get('WLANRSSIDataset')
        TUJI1 = DATASETS.get('TUJI1Dataset')

        assert LongTermWiFi is not None
        assert Tampere is not None
        assert WLANRSSI is not None
        assert TUJI1 is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
