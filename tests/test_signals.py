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


class TestUWBSignal:
    """Tests for UWBSignal class."""

    def test_create_from_dict(self):
        """Test creating UWBSignal from distances dictionary."""
        from indoorloc.signals import UWBSignal

        distances = {'A0': 5.0, 'A1': 7.5, 'A2': 3.2, 'A3': 6.1}
        signal = UWBSignal(distances=distances)

        assert signal.signal_type == 'uwb'
        assert signal.num_anchors == 4
        assert signal.feature_dim == 4

    def test_create_from_anchors(self):
        """Test creating UWBSignal from anchor list."""
        from indoorloc.signals import UWBSignal, UWBAnchor

        anchors = [
            UWBAnchor(anchor_id='A0', distance=5.0, position=(0, 0, 2.5)),
            UWBAnchor(anchor_id='A1', distance=7.5, position=(10, 0, 2.5)),
        ]
        signal = UWBSignal(anchors=anchors)

        assert signal.num_anchors == 2
        assert len(signal.anchor_positions) == 2

    def test_trilateration(self):
        """Test 3D position estimation using trilateration."""
        from indoorloc.signals import UWBSignal, UWBAnchor

        # Create 4 anchors with known positions and distances
        anchors = [
            UWBAnchor(anchor_id='A0', distance=5.0, position=(0, 0, 2.5)),
            UWBAnchor(anchor_id='A1', distance=7.5, position=(10, 0, 2.5)),
            UWBAnchor(anchor_id='A2', distance=6.0, position=(10, 10, 2.5)),
            UWBAnchor(anchor_id='A3', distance=8.0, position=(0, 10, 2.5)),
        ]
        signal = UWBSignal(anchors=anchors)

        position = signal.estimate_position_trilateration()
        assert position is not None
        assert len(position) == 3

    def test_normalize(self):
        """Test UWB signal normalization."""
        from indoorloc.signals import UWBSignal

        distances = {'A0': 5.0, 'A1': 10.0, 'A2': 15.0}
        signal = UWBSignal(distances=distances)

        normalized = signal.normalize(method='minmax')
        data = normalized.to_numpy()

        assert data.min() >= 0.0
        assert data.max() <= 1.0


class TestMagnetometerSignal:
    """Tests for MagnetometerSignal class."""

    def test_create_single_measurement(self):
        """Test creating MagnetometerSignal from single measurement."""
        from indoorloc.signals import MagnetometerSignal

        mag_field = np.array([30.5, 45.2, -12.3], dtype=np.float32)
        signal = MagnetometerSignal(magnetic_field=mag_field)

        assert signal.signal_type == 'magnetometer'
        assert signal.num_samples == 1
        assert signal.feature_dim == 3

    def test_create_time_series(self):
        """Test creating MagnetometerSignal from time series."""
        from indoorloc.signals import MagnetometerSignal

        mag_series = np.random.randn(100, 3).astype(np.float32) * 10 + 40
        signal = MagnetometerSignal(magnetic_field=mag_series, sampling_rate=50.0)

        assert signal.num_samples == 100
        assert signal.sampling_rate == 50.0

    def test_compute_magnitude(self):
        """Test magnetic field magnitude computation."""
        from indoorloc.signals import MagnetometerSignal

        mag_field = np.array([[3, 4, 0]], dtype=np.float32)
        signal = MagnetometerSignal(magnetic_field=mag_field)

        magnitude = signal.compute_magnitude()
        assert np.isclose(magnitude[0], 5.0)

    def test_compute_heading(self):
        """Test heading computation from horizontal components."""
        from indoorloc.signals import MagnetometerSignal

        mag_field = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        signal = MagnetometerSignal(magnetic_field=mag_field)

        heading = signal.compute_heading()
        assert len(heading) == 2

    def test_normalize(self):
        """Test magnetometer signal normalization."""
        from indoorloc.signals import MagnetometerSignal

        mag_field = np.random.randn(50, 3).astype(np.float32) * 10 + 40
        signal = MagnetometerSignal(magnetic_field=mag_field)

        normalized = signal.normalize(method='standard')
        data = normalized.to_numpy()

        assert np.abs(data.mean()) < 0.5
        assert np.abs(data.std() - 1.0) < 0.5


class TestVLCSignal:
    """Tests for VLCSignal class."""

    def test_create_from_arrays(self):
        """Test creating VLCSignal from LED arrays."""
        from indoorloc.signals import VLCSignal

        led_ids = ['LED1', 'LED2', 'LED3', 'LED4']
        received_power = np.array([0.8, 0.5, 0.3, 0.6], dtype=np.float32)

        signal = VLCSignal(led_ids=led_ids, received_power=received_power)

        assert signal.signal_type == 'vlc'
        assert signal.num_leds == 4
        assert signal.feature_dim == 4

    def test_create_from_leds(self):
        """Test creating VLCSignal from LED list."""
        from indoorloc.signals import VLCSignal, LEDTransmitter

        leds = [
            LEDTransmitter(led_id='LED1', received_power=0.8, position=(0, 0, 3)),
            LEDTransmitter(led_id='LED2', received_power=0.5, position=(5, 0, 3)),
        ]
        signal = VLCSignal(leds=leds)

        assert signal.num_leds == 2
        assert len(signal.led_positions) == 2

    def test_distance_estimation(self):
        """Test distance estimation from received power."""
        from indoorloc.signals import VLCSignal

        led_ids = ['LED1']
        received_power = np.array([0.5], dtype=np.float32)
        signal = VLCSignal(led_ids=led_ids, received_power=received_power)

        distance = signal.estimate_distance('LED1')
        assert distance is not None
        assert distance > 0

    def test_normalize(self):
        """Test VLC signal normalization."""
        from indoorloc.signals import VLCSignal

        led_ids = ['LED1', 'LED2', 'LED3']
        received_power = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        signal = VLCSignal(led_ids=led_ids, received_power=received_power)

        normalized = signal.normalize(method='minmax')
        data = normalized.received_power

        assert data.min() >= 0.0
        assert data.max() <= 1.0


class TestUltrasoundSignal:
    """Tests for UltrasoundSignal class."""

    def test_create_from_tof(self):
        """Test creating UltrasoundSignal from TOF measurements."""
        from indoorloc.signals import UltrasoundSignal

        tof = {
            'US1': 0.0145,
            'US2': 0.0218,
            'US3': 0.0093,
            'US4': 0.0178,
        }
        signal = UltrasoundSignal(tof_measurements=tof)

        assert signal.signal_type == 'ultrasound'
        assert signal.num_transmitters == 4
        assert signal.feature_dim == 4

    def test_tof_to_distance_conversion(self):
        """Test TOF to distance conversion."""
        from indoorloc.signals import UltrasoundSignal

        tof = {'US1': 0.01}  # 0.01 seconds
        signal = UltrasoundSignal(tof_measurements=tof)

        distance = signal.tof_to_distance(0.01)
        expected = 0.01 * 343.0  # speed of sound
        assert np.isclose(distance, expected)

    def test_temperature_compensation(self):
        """Test speed of sound adjustment for temperature."""
        from indoorloc.signals import UltrasoundSignal

        tof = {'US1': 0.01}

        # Signal at 20°C
        signal_20c = UltrasoundSignal(tof_measurements=tof, temperature=20.0)

        # Signal at 30°C
        signal_30c = UltrasoundSignal(tof_measurements=tof, temperature=30.0)

        # Speed of sound should be higher at higher temperature
        assert signal_30c.speed_of_sound > signal_20c.speed_of_sound

    def test_normalize(self):
        """Test ultrasound signal normalization."""
        from indoorloc.signals import UltrasoundSignal

        tof = {'US1': 0.01, 'US2': 0.02, 'US3': 0.03}
        signal = UltrasoundSignal(tof_measurements=tof)

        normalized = signal.normalize(method='minmax')
        data = normalized.to_numpy()

        assert data.min() >= 0.0
        assert data.max() <= 1.0


class TestHybridSignal:
    """Tests for HybridSignal class."""

    def test_create_hybrid_signal(self):
        """Test creating HybridSignal from multiple sensors."""
        from indoorloc.signals import HybridSignal, WiFiSignal, MagnetometerSignal

        wifi = WiFiSignal(rssi_values=np.random.randn(100).astype(np.float32))
        mag = MagnetometerSignal(
            magnetic_field=np.random.randn(10, 3).astype(np.float32)
        )

        hybrid = HybridSignal(sensors={'wifi': wifi, 'magnetometer': mag})

        assert hybrid.signal_type == 'hybrid'
        assert hybrid.num_modalities == 2
        assert 'wifi' in hybrid.signal_types
        assert 'magnetometer' in hybrid.signal_types

    def test_get_signal(self):
        """Test retrieving individual signals from hybrid."""
        from indoorloc.signals import HybridSignal, WiFiSignal, UWBSignal

        wifi = WiFiSignal(rssi_values=np.random.randn(50).astype(np.float32))
        uwb = UWBSignal(distances={'A0': 5.0, 'A1': 7.5, 'A2': 3.2})

        hybrid = HybridSignal(sensors={'wifi': wifi, 'uwb': uwb})

        retrieved_wifi = hybrid.get_signal('wifi')
        assert retrieved_wifi is not None
        assert retrieved_wifi.signal_type == 'wifi'

    def test_feature_dim(self):
        """Test total feature dimension of hybrid signal."""
        from indoorloc.signals import HybridSignal, WiFiSignal, MagnetometerSignal

        wifi = WiFiSignal(rssi_values=np.zeros(100, dtype=np.float32))
        mag = MagnetometerSignal(
            magnetic_field=np.zeros((50, 3), dtype=np.float32)
        )

        hybrid = HybridSignal(sensors={'wifi': wifi, 'magnetometer': mag})

        # Total dim = 100 (wifi) + 150 (mag 50*3)
        assert hybrid.feature_dim == wifi.feature_dim + mag.feature_dim

    def test_to_tensor(self):
        """Test hybrid signal to tensor conversion."""
        from indoorloc.signals import HybridSignal, WiFiSignal, MagnetometerSignal

        wifi = WiFiSignal(rssi_values=np.random.randn(50).astype(np.float32))
        mag = MagnetometerSignal(magnetic_field=np.random.randn(10, 3).astype(np.float32))

        hybrid = HybridSignal(sensors={'wifi': wifi, 'magnetometer': mag})

        tensor = hybrid.to_tensor()
        # 50 (wifi) + 30 (mag 10x3) = 80
        assert tensor.shape[0] == wifi.feature_dim + mag.feature_dim

    def test_fusion_weights(self):
        """Test fusion weights in hybrid signal."""
        from indoorloc.signals import HybridSignal, WiFiSignal, MagnetometerSignal

        wifi = WiFiSignal(rssi_values=np.random.randn(50).astype(np.float32))
        mag = MagnetometerSignal(magnetic_field=np.random.randn(10, 3).astype(np.float32))

        weights = {'wifi': 0.7, 'magnetometer': 0.3}
        hybrid = HybridSignal(sensors={'wifi': wifi, 'magnetometer': mag}, fusion_weights=weights)

        # Weights should be normalized to sum to 1
        total_weight = sum(hybrid.fusion_weights.values())
        assert np.isclose(total_weight, 1.0)

    def test_normalize(self):
        """Test normalizing all signals in hybrid."""
        from indoorloc.signals import HybridSignal, WiFiSignal, MagnetometerSignal

        wifi = WiFiSignal(rssi_values=np.random.randn(50).astype(np.float32) * 10)
        mag = MagnetometerSignal(magnetic_field=np.random.randn(10, 3).astype(np.float32) * 10)

        hybrid = HybridSignal(sensors={'wifi': wifi, 'magnetometer': mag})
        normalized = hybrid.normalize(method='standard')

        assert normalized.num_modalities == 2
        assert normalized.signal_type == 'hybrid'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
