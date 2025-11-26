"""Tests for UWB datasets (Phase 4)."""
import pytest


class TestUWBIndoorDataset:
    """Tests for UWB Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing UWBIndoor dataset class."""
        from indoorloc.datasets import UWBIndoorDataset, UWBIndoor

        assert UWBIndoorDataset is not None
        assert UWBIndoor is not None
        assert UWBIndoor == UWBIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import UWBIndoorDataset

        assert hasattr(UWBIndoorDataset, 'ZENODO_RECORD_ID')
        assert hasattr(UWBIndoorDataset, 'MAX_DISTANCE')
        assert UWBIndoorDataset.ZENODO_RECORD_ID == '5789876'
        assert UWBIndoorDataset.MAX_DISTANCE == 100.0

    def test_dataset_properties(self):
        """Test dataset properties."""
        from indoorloc.datasets import UWBIndoorDataset

        # Test num_anchors property
        dataset = UWBIndoorDataset.__new__(UWBIndoorDataset)
        dataset._num_anchors = None
        assert dataset.num_anchors == 0

        dataset._num_anchors = 5
        assert dataset.num_anchors == 5

    def test_anchor_positions_property(self):
        """Test anchor_positions property."""
        from indoorloc.datasets import UWBIndoorDataset
        import numpy as np

        dataset = UWBIndoorDataset.__new__(UWBIndoorDataset)
        dataset._anchor_positions = {
            'A1': np.array([0.0, 0.0, 2.5]),
            'A2': np.array([10.0, 0.0, 2.5]),
        }

        positions = dataset.anchor_positions
        assert len(positions) == 2
        assert 'A1' in positions
        assert 'A2' in positions


class TestUWBRangingDataset:
    """Tests for UWB Ranging dataset."""

    def test_dataset_class_import(self):
        """Test importing UWBRanging dataset class."""
        from indoorloc.datasets import UWBRangingDataset, UWBRanging

        assert UWBRangingDataset is not None
        assert UWBRanging is not None
        assert UWBRanging == UWBRangingDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import UWBRangingDataset

        assert hasattr(UWBRangingDataset, 'BASE_URL')
        assert hasattr(UWBRangingDataset, 'MAX_DISTANCE')
        assert hasattr(UWBRangingDataset, 'NOT_DETECTED_RSSI')
        assert UWBRangingDataset.MAX_DISTANCE == 100.0
        assert UWBRangingDataset.NOT_DETECTED_RSSI == -100.0

    def test_dataset_properties(self):
        """Test dataset properties."""
        from indoorloc.datasets import UWBRangingDataset

        # Test num_anchors property
        dataset = UWBRangingDataset.__new__(UWBRangingDataset)
        dataset._num_anchors = None
        assert dataset.num_anchors == 0

        dataset._num_anchors = 8
        assert dataset.num_anchors == 8

    def test_anchor_positions_property(self):
        """Test anchor_positions property."""
        from indoorloc.datasets import UWBRangingDataset
        import numpy as np

        dataset = UWBRangingDataset.__new__(UWBRangingDataset)
        dataset._anchor_positions = {
            'A1': np.array([0.0, 0.0, 3.0]),
            'A2': np.array([5.0, 5.0, 3.0]),
        }

        positions = dataset.anchor_positions
        assert len(positions) == 2
        assert 'A1' in positions


class TestUWBDatasetRegistry:
    """Tests for UWB dataset registry integration."""

    def test_all_uwb_datasets_registered(self):
        """Test that all UWB datasets are registered."""
        from indoorloc.registry import DATASETS

        assert 'UWBIndoorDataset' in DATASETS._module_dict
        assert 'UWBRangingDataset' in DATASETS._module_dict

    def test_uwb_dataset_retrieval_from_registry(self):
        """Test retrieving UWB datasets from registry."""
        from indoorloc.registry import DATASETS

        UWBIndoor = DATASETS.get('UWBIndoorDataset')
        UWBRanging = DATASETS.get('UWBRangingDataset')

        assert UWBIndoor is not None
        assert UWBRanging is not None

    def test_uwb_datasets_inherit_from_uwb_dataset(self):
        """Test that UWB datasets inherit from UWBDataset base class."""
        from indoorloc.datasets import (
            UWBIndoorDataset,
            UWBRangingDataset,
            UWBDataset
        )

        assert issubclass(UWBIndoorDataset, UWBDataset)
        assert issubclass(UWBRangingDataset, UWBDataset)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
