"""Tests for final batch datasets (Phase 6)."""
import pytest


class TestRSSBasedDataset:
    """Tests for RSS-Based dataset."""

    def test_dataset_class_import(self):
        """Test importing RSSBased dataset class."""
        from indoorloc.datasets import RSSBasedDataset, RSSBased

        assert RSSBasedDataset is not None
        assert RSSBased is not None
        assert RSSBased == RSSBasedDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import RSSBasedDataset

        assert hasattr(RSSBasedDataset, 'ZENODO_RECORD_ID')
        assert hasattr(RSSBasedDataset, 'NOT_DETECTED_VALUE')
        assert RSSBasedDataset.ZENODO_RECORD_ID == '5678901'
        assert RSSBasedDataset.NOT_DETECTED_VALUE == 100

    def test_dataset_inherits_from_wifi_dataset(self):
        """Test that RSSBased inherits from WiFiDataset."""
        from indoorloc.datasets import RSSBasedDataset, WiFiDataset

        assert issubclass(RSSBasedDataset, WiFiDataset)


class TestCSIIndoorDataset:
    """Tests for CSI Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing CSIIndoor dataset class."""
        from indoorloc.datasets import CSIIndoorDataset, CSIIndoor

        assert CSIIndoorDataset is not None
        assert CSIIndoor is not None
        assert CSIIndoor == CSIIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import CSIIndoorDataset

        assert hasattr(CSIIndoorDataset, 'BASE_URL')
        assert hasattr(CSIIndoorDataset, 'NOT_DETECTED_VALUE')
        assert hasattr(CSIIndoorDataset, 'NUM_SUBCARRIERS')
        assert CSIIndoorDataset.NOT_DETECTED_VALUE == -80.0
        assert CSIIndoorDataset.NUM_SUBCARRIERS == 52

    def test_dataset_inherits_from_wifi_dataset(self):
        """Test that CSIIndoor inherits from WiFiDataset."""
        from indoorloc.datasets import CSIIndoorDataset, WiFiDataset

        assert issubclass(CSIIndoorDataset, WiFiDataset)


class TestRFIDIndoorDataset:
    """Tests for RFID Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing RFIDIndoor dataset class."""
        from indoorloc.datasets import RFIDIndoorDataset, RFIDIndoor

        assert RFIDIndoorDataset is not None
        assert RFIDIndoor is not None
        assert RFIDIndoor == RFIDIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import RFIDIndoorDataset

        assert hasattr(RFIDIndoorDataset, 'UCI_DATASET_NAME')
        assert hasattr(RFIDIndoorDataset, 'REQUIRED_FILES')
        assert RFIDIndoorDataset.UCI_DATASET_NAME == 'rfid-indoor-localization'

    def test_dataset_inherits_from_base_dataset(self):
        """Test that RFIDIndoor inherits from BaseDataset."""
        from indoorloc.datasets import RFIDIndoorDataset, BaseDataset

        assert issubclass(RFIDIndoorDataset, BaseDataset)


class TestFinalDatasetRegistry:
    """Tests for final dataset registry integration."""

    def test_all_final_datasets_registered(self):
        """Test that all final datasets are registered."""
        from indoorloc.registry import DATASETS

        assert 'RSSBasedDataset' in DATASETS._module_dict
        assert 'CSIIndoorDataset' in DATASETS._module_dict
        assert 'RFIDIndoorDataset' in DATASETS._module_dict

    def test_final_dataset_retrieval_from_registry(self):
        """Test retrieving final datasets from registry."""
        from indoorloc.registry import DATASETS

        RSSBased = DATASETS.get('RSSBasedDataset')
        CSIIndoor = DATASETS.get('CSIIndoorDataset')
        RFIDIndoor = DATASETS.get('RFIDIndoorDataset')

        assert RSSBased is not None
        assert CSIIndoor is not None
        assert RFIDIndoor is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
