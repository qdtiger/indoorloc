"""Tests for other signal type datasets (Phase 5)."""
import pytest


class TestMagneticIndoorDataset:
    """Tests for Magnetic Field Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing MagneticIndoor dataset class."""
        from indoorloc.datasets import MagneticIndoorDataset, MagneticIndoor

        assert MagneticIndoorDataset is not None
        assert MagneticIndoor is not None
        assert MagneticIndoor == MagneticIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import MagneticIndoorDataset

        assert hasattr(MagneticIndoorDataset, 'ZENODO_RECORD_ID')
        assert hasattr(MagneticIndoorDataset, 'MIN_FIELD')
        assert hasattr(MagneticIndoorDataset, 'MAX_FIELD')
        assert MagneticIndoorDataset.ZENODO_RECORD_ID == '4321098'
        assert MagneticIndoorDataset.MIN_FIELD == -100.0
        assert MagneticIndoorDataset.MAX_FIELD == 100.0


class TestVLCIndoorDataset:
    """Tests for VLC Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing VLCIndoor dataset class."""
        from indoorloc.datasets import VLCIndoorDataset, VLCIndoor

        assert VLCIndoorDataset is not None
        assert VLCIndoor is not None
        assert VLCIndoor == VLCIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import VLCIndoorDataset

        assert hasattr(VLCIndoorDataset, 'BASE_URL')
        assert hasattr(VLCIndoorDataset, 'NOT_DETECTED_RSSI')
        assert hasattr(VLCIndoorDataset, 'MIN_SNR')
        assert hasattr(VLCIndoorDataset, 'MAX_SNR')
        assert VLCIndoorDataset.NOT_DETECTED_RSSI == -100.0
        assert VLCIndoorDataset.MIN_SNR == -20.0
        assert VLCIndoorDataset.MAX_SNR == 40.0

    def test_dataset_properties(self):
        """Test dataset properties."""
        from indoorloc.datasets import VLCIndoorDataset

        # Test num_transmitters property
        dataset = VLCIndoorDataset.__new__(VLCIndoorDataset)
        dataset._num_transmitters = None
        assert dataset.num_transmitters == 0

        dataset._num_transmitters = 6
        assert dataset.num_transmitters == 6

    def test_led_positions_property(self):
        """Test led_positions property."""
        from indoorloc.datasets import VLCIndoorDataset
        import numpy as np

        dataset = VLCIndoorDataset.__new__(VLCIndoorDataset)
        dataset._led_positions = {
            'LED1': np.array([0.0, 0.0, 3.0]),
            'LED2': np.array([5.0, 5.0, 3.0]),
        }

        positions = dataset.led_positions
        assert len(positions) == 2
        assert 'LED1' in positions
        assert 'LED2' in positions


class TestUltrasoundIndoorDataset:
    """Tests for Ultrasound Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing UltrasoundIndoor dataset class."""
        from indoorloc.datasets import UltrasoundIndoorDataset, UltrasoundIndoor

        assert UltrasoundIndoorDataset is not None
        assert UltrasoundIndoor is not None
        assert UltrasoundIndoor == UltrasoundIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import UltrasoundIndoorDataset

        assert hasattr(UltrasoundIndoorDataset, 'UCI_DATASET_NAME')
        assert hasattr(UltrasoundIndoorDataset, 'SPEED_OF_SOUND')
        assert hasattr(UltrasoundIndoorDataset, 'MAX_DISTANCE')
        assert UltrasoundIndoorDataset.UCI_DATASET_NAME == 'ultrasound-indoor-localization'
        assert UltrasoundIndoorDataset.SPEED_OF_SOUND == 343.0
        assert UltrasoundIndoorDataset.MAX_DISTANCE == 20.0

    def test_dataset_properties(self):
        """Test dataset properties."""
        from indoorloc.datasets import UltrasoundIndoorDataset

        # Test num_beacons property
        dataset = UltrasoundIndoorDataset.__new__(UltrasoundIndoorDataset)
        dataset._num_beacons = None
        assert dataset.num_beacons == 0

        dataset._num_beacons = 4
        assert dataset.num_beacons == 4


class TestOtherSignalDatasetRegistry:
    """Tests for other signal dataset registry integration."""

    def test_all_other_signal_datasets_registered(self):
        """Test that all other signal datasets are registered."""
        from indoorloc.registry import DATASETS

        assert 'MagneticIndoorDataset' in DATASETS._module_dict
        assert 'VLCIndoorDataset' in DATASETS._module_dict
        assert 'UltrasoundIndoorDataset' in DATASETS._module_dict

    def test_other_signal_dataset_retrieval_from_registry(self):
        """Test retrieving other signal datasets from registry."""
        from indoorloc.registry import DATASETS

        MagneticIndoor = DATASETS.get('MagneticIndoorDataset')
        VLCIndoor = DATASETS.get('VLCIndoorDataset')
        UltrasoundIndoor = DATASETS.get('UltrasoundIndoorDataset')

        assert MagneticIndoor is not None
        assert VLCIndoor is not None
        assert UltrasoundIndoor is not None

    def test_magnetic_dataset_inherits_from_magnetic_dataset(self):
        """Test that Magnetic dataset inherits from MagneticDataset base class."""
        from indoorloc.datasets import (
            MagneticIndoorDataset,
            MagneticDataset
        )

        assert issubclass(MagneticIndoorDataset, MagneticDataset)

    def test_vlc_and_ultrasound_inherit_from_base_dataset(self):
        """Test that VLC and Ultrasound datasets inherit from BaseDataset."""
        from indoorloc.datasets import (
            VLCIndoorDataset,
            UltrasoundIndoorDataset,
            BaseDataset
        )

        assert issubclass(VLCIndoorDataset, BaseDataset)
        assert issubclass(UltrasoundIndoorDataset, BaseDataset)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
