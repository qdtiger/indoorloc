"""Tests for localizer classes."""
import pytest
import numpy as np


class TestKNNLocalizer:
    """Tests for KNNLocalizer class."""

    def test_create_model(self):
        """Test creating KNN model."""
        import indoorloc as iloc

        model = iloc.create_model('KNNLocalizer', k=5)
        assert model is not None
        assert not model.is_trained

    def test_train_and_predict(self):
        """Test training and prediction."""
        import indoorloc as iloc

        # Create synthetic data
        np.random.seed(42)
        num_samples = 100
        num_aps = 50

        train_signals = [
            iloc.WiFiSignal(rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32))
            for _ in range(num_samples)
        ]
        train_locations = [
            iloc.Location.from_coordinates(
                x=np.random.uniform(0, 100),
                y=np.random.uniform(0, 100),
                floor=np.random.randint(0, 3)
            )
            for _ in range(num_samples)
        ]

        # Train
        model = iloc.create_model('KNNLocalizer', k=3)
        model.fit(train_signals, train_locations)

        assert model.is_trained

        # Predict
        test_signal = iloc.WiFiSignal(
            rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32)
        )
        result = model.predict(test_signal)

        assert result is not None
        assert hasattr(result, 'location')
        assert result.location.coordinate.x >= 0
        assert result.location.coordinate.y >= 0

    def test_batch_predict(self):
        """Test batch prediction."""
        import indoorloc as iloc

        np.random.seed(42)
        num_train = 100
        num_test = 20
        num_aps = 50

        train_signals = [
            iloc.WiFiSignal(rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32))
            for _ in range(num_train)
        ]
        train_locations = [
            iloc.Location.from_coordinates(x=np.random.uniform(0, 100), y=np.random.uniform(0, 100))
            for _ in range(num_train)
        ]

        model = iloc.create_model('KNNLocalizer', k=3)
        model.fit(train_signals, train_locations)

        test_signals = [
            iloc.WiFiSignal(rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32))
            for _ in range(num_test)
        ]

        results = model.predict_batch(test_signals)

        assert len(results) == num_test
        assert all(r is not None for r in results)

    def test_save_and_load(self):
        """Test model save and load."""
        import indoorloc as iloc
        import tempfile
        import os

        np.random.seed(42)
        num_samples = 50
        num_aps = 30

        train_signals = [
            iloc.WiFiSignal(rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32))
            for _ in range(num_samples)
        ]
        train_locations = [
            iloc.Location.from_coordinates(x=np.random.uniform(0, 100), y=np.random.uniform(0, 100))
            for _ in range(num_samples)
        ]

        model = iloc.create_model('KNNLocalizer', k=3)
        model.fit(train_signals, train_locations)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name

        try:
            model.save(model_path)

            # Load
            loaded_model = iloc.create_model('KNNLocalizer')
            loaded_model.load(model_path)

            assert loaded_model.is_trained

            # Verify predictions match
            test_signal = iloc.WiFiSignal(
                rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32)
            )

            result1 = model.predict(test_signal)
            result2 = loaded_model.predict(test_signal)

            assert result1.x == pytest.approx(result2.x, rel=1e-5)
            assert result1.y == pytest.approx(result2.y, rel=1e-5)
        finally:
            os.unlink(model_path)


class TestWKNNLocalizer:
    """Tests for Weighted KNN Localizer."""

    def test_weighted_knn(self):
        """Test weighted KNN prediction."""
        import indoorloc as iloc

        np.random.seed(42)
        num_samples = 100
        num_aps = 50

        train_signals = [
            iloc.WiFiSignal(rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32))
            for _ in range(num_samples)
        ]
        train_locations = [
            iloc.Location.from_coordinates(
                x=np.random.uniform(0, 100),
                y=np.random.uniform(0, 100),
                floor=np.random.randint(0, 3)
            )
            for _ in range(num_samples)
        ]

        model = iloc.create_model('WKNNLocalizer', k=5)
        model.fit(train_signals, train_locations)

        test_signal = iloc.WiFiSignal(
            rssi_values=np.random.uniform(-100, -30, num_aps).astype(np.float32)
        )
        result = model.predict(test_signal)

        assert result is not None
        assert result.location.coordinate.x >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
