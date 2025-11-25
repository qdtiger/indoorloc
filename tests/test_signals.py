"""Tests for signal classes."""
import pytest
import numpy as np


class TestWiFiSignal:
    """Tests for WiFiSignal class."""

    def test_create_from_array(self):
        """Test creating WiFiSignal from numpy array."""
        from indoorloc.signals import WiFiSignal

        rssi = np.random.uniform(-100, -30, 520).astype(np.float32)
        signal = WiFiSignal(rssi_values=rssi)

        assert signal.signal_type == 'wifi'
        assert len(signal.rssi_values) == 520
        assert signal.num_aps == 520

    def test_sparse_mode(self):
        """Test sparse mode with AP info."""
        from indoorloc.signals import WiFiSignal, APInfo

        ap_list = [
            APInfo(bssid='aa:bb:cc:dd:ee:01', rssi=-65),
            APInfo(bssid='aa:bb:cc:dd:ee:02', rssi=-72),
        ]
        signal = WiFiSignal(ap_list=ap_list)

        assert signal.is_sparse
        assert signal.num_detected == 2

    def test_normalize_minmax(self):
        """Test minmax normalization."""
        from indoorloc.signals import WiFiSignal

        rssi = np.array([-100, -50, 100, -30], dtype=np.float32)  # 100 = not detected
        signal = WiFiSignal(rssi_values=rssi)

        normalized = signal.normalize(method='minmax')

        # Check normalized values are in [0, 1]
        mask = signal.rssi_values != 100
        assert np.all(normalized.rssi_values[mask] >= 0)
        assert np.all(normalized.rssi_values[mask] <= 1)
        # Not detected should be 0
        assert normalized.rssi_values[2] == 0.0

    def test_to_tensor(self):
        """Test conversion to tensor."""
        from indoorloc.signals import WiFiSignal

        rssi = np.random.uniform(-100, -30, 100).astype(np.float32)
        signal = WiFiSignal(rssi_values=rssi)

        tensor = signal.to_tensor()
        assert tensor.shape == (100,)
        assert tensor.dtype.name.startswith('float')


class TestBLESignal:
    """Tests for BLESignal class."""

    def test_create_from_beacons(self):
        """Test creating BLESignal from beacon list."""
        from indoorloc.signals import BLESignal, BLEBeacon

        beacons = [
            BLEBeacon(uuid='test-uuid', major=1, minor=1, rssi=-65),
            BLEBeacon(uuid='test-uuid', major=1, minor=2, rssi=-72),
        ]
        signal = BLESignal(beacons=beacons)

        assert signal.signal_type == 'ble'
        assert signal.is_sparse
        assert signal.num_beacons == 2

    def test_create_from_array(self):
        """Test creating BLESignal from RSSI array."""
        from indoorloc.signals import BLESignal

        rssi = np.array([-65, -72, -100, -100], dtype=np.float32)
        signal = BLESignal(rssi_values=rssi)

        assert not signal.is_sparse
        assert signal.num_beacons == 2  # Only non -100 values

    def test_beacon_distance_estimation(self):
        """Test distance estimation from beacon."""
        from indoorloc.signals import BLEBeacon

        beacon = BLEBeacon(rssi=-65, tx_power=-59)
        distance = beacon.estimated_distance()

        assert distance is not None
        assert distance > 0


class TestIMUSignal:
    """Tests for IMUSignal class."""

    def test_create_from_arrays(self):
        """Test creating IMUSignal from numpy arrays."""
        from indoorloc.signals import IMUSignal

        accel = np.random.randn(100, 3).astype(np.float32)
        gyro = np.random.randn(100, 3).astype(np.float32)

        signal = IMUSignal(accelerometer=accel, gyroscope=gyro, sampling_rate=100)

        assert signal.signal_type == 'imu'
        assert signal.num_samples == 100
        assert signal.sampling_rate == 100

    def test_create_from_readings(self):
        """Test creating IMUSignal from readings list."""
        from indoorloc.signals import IMUSignal, IMUReading

        readings = [
            IMUReading(timestamp=i/100, accelerometer=(0, 0, 9.8), gyroscope=(0, 0, 0))
            for i in range(100)
        ]
        signal = IMUSignal(readings=readings)

        assert signal.num_samples == 100
        assert signal.accelerometer is not None
        assert signal.gyroscope is not None

    def test_get_window(self):
        """Test extracting window from IMU signal."""
        from indoorloc.signals import IMUSignal

        accel = np.random.randn(100, 3).astype(np.float32)
        gyro = np.random.randn(100, 3).astype(np.float32)
        signal = IMUSignal(accelerometer=accel, gyroscope=gyro)

        window = signal.get_window(10, 20)
        assert window.num_samples == 20

    def test_normalize(self):
        """Test IMU signal normalization."""
        from indoorloc.signals import IMUSignal

        accel = np.random.randn(100, 3).astype(np.float32) * 10
        gyro = np.random.randn(100, 3).astype(np.float32)
        signal = IMUSignal(accelerometer=accel, gyroscope=gyro)

        normalized = signal.normalize(method='standard')

        # Check normalized accelerometer has ~0 mean and ~1 std
        assert np.abs(normalized.accelerometer.mean()) < 0.1
        assert np.abs(normalized.accelerometer.std() - 1.0) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
