"""Tests for Hybrid datasets (Phase 3)."""
import pytest


class TestWiFiIMUHybridDataset:
    """Tests for WiFi+IMU Hybrid dataset."""

    def test_dataset_class_import(self):
        """Test importing WiFiIMUHybrid dataset class."""
        from indoorloc.datasets import WiFiIMUHybridDataset, WiFiIMUHybrid

        assert WiFiIMUHybridDataset is not None
        assert WiFiIMUHybrid is not None
        assert WiFiIMUHybrid == WiFiIMUHybridDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import WiFiIMUHybridDataset

        assert hasattr(WiFiIMUHybridDataset, 'ZENODO_RECORD_ID')
        assert hasattr(WiFiIMUHybridDataset, 'NOT_DETECTED_VALUE')
        assert WiFiIMUHybridDataset.ZENODO_RECORD_ID == '3932395'

    def test_dataset_signal_types(self):
        """Test signal_types property."""
        from indoorloc.datasets import WiFiIMUHybridDataset

        # Test with all signals enabled
        dataset_all = WiFiIMUHybridDataset.__new__(WiFiIMUHybridDataset)
        dataset_all.use_wifi = True
        dataset_all.use_imu = True
        assert 'wifi' in dataset_all.signal_types
        assert 'imu' in dataset_all.signal_types

        # Test with only WiFi
        dataset_wifi = WiFiIMUHybridDataset.__new__(WiFiIMUHybridDataset)
        dataset_wifi.use_wifi = True
        dataset_wifi.use_imu = False
        assert 'wifi' in dataset_wifi.signal_types
        assert 'imu' not in dataset_wifi.signal_types


class TestWiFiMagneticHybridDataset:
    """Tests for WiFi+Magnetic Hybrid dataset."""

    def test_dataset_class_import(self):
        """Test importing WiFiMagneticHybrid dataset class."""
        from indoorloc.datasets import WiFiMagneticHybridDataset, WiFiMagneticHybrid

        assert WiFiMagneticHybridDataset is not None
        assert WiFiMagneticHybrid is not None
        assert WiFiMagneticHybrid == WiFiMagneticHybridDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import WiFiMagneticHybridDataset

        assert hasattr(WiFiMagneticHybridDataset, 'UCI_DATASET_NAME')
        assert hasattr(WiFiMagneticHybridDataset, 'NOT_DETECTED_VALUE')
        assert WiFiMagneticHybridDataset.UCI_DATASET_NAME == 'wifi-magnetic-indoor-localization'

    def test_dataset_signal_types(self):
        """Test signal_types property."""
        from indoorloc.datasets import WiFiMagneticHybridDataset

        # Test with all signals enabled
        dataset_all = WiFiMagneticHybridDataset.__new__(WiFiMagneticHybridDataset)
        dataset_all.use_wifi = True
        dataset_all.use_magnetic = True
        assert 'wifi' in dataset_all.signal_types
        assert 'magnetic' in dataset_all.signal_types


class TestMultiModalIndoorDataset:
    """Tests for Multi-modal Indoor dataset."""

    def test_dataset_class_import(self):
        """Test importing MultiModalIndoor dataset class."""
        from indoorloc.datasets import MultiModalIndoorDataset, MultiModalIndoor

        assert MultiModalIndoorDataset is not None
        assert MultiModalIndoor is not None
        assert MultiModalIndoor == MultiModalIndoorDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import MultiModalIndoorDataset

        assert hasattr(MultiModalIndoorDataset, 'BASE_URL')
        assert hasattr(MultiModalIndoorDataset, 'NOT_DETECTED_VALUE')
        assert hasattr(MultiModalIndoorDataset, 'BLE_NOT_DETECTED')

    def test_dataset_signal_types(self):
        """Test signal_types property."""
        from indoorloc.datasets import MultiModalIndoorDataset

        # Test with all signals enabled
        dataset_all = MultiModalIndoorDataset.__new__(MultiModalIndoorDataset)
        dataset_all.use_wifi = True
        dataset_all.use_ble = True
        dataset_all.use_imu = True
        assert 'wifi' in dataset_all.signal_types
        assert 'ble' in dataset_all.signal_types
        assert 'imu' in dataset_all.signal_types


class TestSensorFusionDataset:
    """Tests for Sensor Fusion dataset."""

    def test_dataset_class_import(self):
        """Test importing SensorFusion dataset class."""
        from indoorloc.datasets import SensorFusionDataset, SensorFusion

        assert SensorFusionDataset is not None
        assert SensorFusion is not None
        assert SensorFusion == SensorFusionDataset

    def test_dataset_constants(self):
        """Test dataset constants."""
        from indoorloc.datasets import SensorFusionDataset

        assert hasattr(SensorFusionDataset, 'ZENODO_RECORD_ID')
        assert hasattr(SensorFusionDataset, 'NOT_DETECTED_VALUE')
        assert hasattr(SensorFusionDataset, 'BLE_NOT_DETECTED')
        assert SensorFusionDataset.ZENODO_RECORD_ID == '4567890'

    def test_dataset_signal_types(self):
        """Test signal_types property."""
        from indoorloc.datasets import SensorFusionDataset

        # Test with all signals enabled
        dataset_all = SensorFusionDataset.__new__(SensorFusionDataset)
        dataset_all.use_wifi = True
        dataset_all.use_ble = True
        dataset_all.use_magnetic = True
        assert 'wifi' in dataset_all.signal_types
        assert 'ble' in dataset_all.signal_types
        assert 'magnetic' in dataset_all.signal_types


class TestHybridDatasetRegistry:
    """Tests for Hybrid dataset registry integration."""

    def test_all_hybrid_datasets_registered(self):
        """Test that all Hybrid datasets are registered."""
        from indoorloc.registry import DATASETS

        assert 'WiFiIMUHybridDataset' in DATASETS._module_dict
        assert 'WiFiMagneticHybridDataset' in DATASETS._module_dict
        assert 'MultiModalIndoorDataset' in DATASETS._module_dict
        assert 'SensorFusionDataset' in DATASETS._module_dict

    def test_hybrid_dataset_retrieval_from_registry(self):
        """Test retrieving Hybrid datasets from registry."""
        from indoorloc.registry import DATASETS

        WiFiIMUHybrid = DATASETS.get('WiFiIMUHybridDataset')
        WiFiMagneticHybrid = DATASETS.get('WiFiMagneticHybridDataset')
        MultiModalIndoor = DATASETS.get('MultiModalIndoorDataset')
        SensorFusion = DATASETS.get('SensorFusionDataset')

        assert WiFiIMUHybrid is not None
        assert WiFiMagneticHybrid is not None
        assert MultiModalIndoor is not None
        assert SensorFusion is not None

    def test_hybrid_datasets_inherit_from_hybrid_dataset(self):
        """Test that Hybrid datasets inherit from HybridDataset base class."""
        from indoorloc.datasets import (
            WiFiIMUHybridDataset,
            WiFiMagneticHybridDataset,
            MultiModalIndoorDataset,
            SensorFusionDataset,
            HybridDataset
        )

        assert issubclass(WiFiIMUHybridDataset, HybridDataset)
        assert issubclass(WiFiMagneticHybridDataset, HybridDataset)
        assert issubclass(MultiModalIndoorDataset, HybridDataset)
        assert issubclass(SensorFusionDataset, HybridDataset)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
