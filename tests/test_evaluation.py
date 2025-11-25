"""Tests for evaluation metrics."""
import pytest
import numpy as np


class TestMetrics:
    """Tests for evaluation metrics."""

    def setup_method(self):
        """Setup test fixtures."""
        from indoorloc.locations import Location

        # Create mock predictions and ground truths
        self.predictions = [
            Location.from_coordinates(x=10, y=10, floor=0, building_id='0'),
            Location.from_coordinates(x=25, y=25, floor=1, building_id='0'),
            Location.from_coordinates(x=55, y=55, floor=2, building_id='1'),
            Location.from_coordinates(x=80, y=80, floor=3, building_id='1'),
        ]

        self.ground_truths = [
            Location.from_coordinates(x=12, y=12, floor=0, building_id='0'),
            Location.from_coordinates(x=20, y=20, floor=1, building_id='0'),
            Location.from_coordinates(x=50, y=50, floor=2, building_id='0'),  # Wrong building
            Location.from_coordinates(x=75, y=75, floor=2, building_id='1'),  # Wrong floor
        ]

    def test_mean_position_error(self):
        """Test MeanPositionError metric."""
        from indoorloc.evaluation import MeanPositionError

        metric = MeanPositionError()
        result = metric.compute(self.predictions, self.ground_truths)

        assert isinstance(result, float)
        assert result > 0

    def test_median_position_error(self):
        """Test MedianPositionError metric."""
        from indoorloc.evaluation import MedianPositionError

        metric = MedianPositionError()
        result = metric.compute(self.predictions, self.ground_truths)

        assert isinstance(result, float)
        assert result > 0

    def test_percentile_error(self):
        """Test PercentileError metric."""
        from indoorloc.evaluation import PercentileError

        metric_75 = PercentileError(percentile=75)
        metric_95 = PercentileError(percentile=95)

        result_75 = metric_75.compute(self.predictions, self.ground_truths)
        result_95 = metric_95.compute(self.predictions, self.ground_truths)

        assert result_75 <= result_95  # 95th percentile should be >= 75th

    def test_floor_accuracy(self):
        """Test FloorAccuracy metric."""
        from indoorloc.evaluation import FloorAccuracy

        metric = FloorAccuracy()
        result = metric.compute(self.predictions, self.ground_truths)

        # 3 out of 4 correct floors
        assert result == pytest.approx(75.0)

    def test_building_accuracy(self):
        """Test BuildingAccuracy metric."""
        from indoorloc.evaluation import BuildingAccuracy

        metric = BuildingAccuracy()
        result = metric.compute(self.predictions, self.ground_truths)

        # 3 out of 4 correct buildings
        assert result == pytest.approx(75.0)

    def test_cdf_analysis(self):
        """Test CDFAnalysis metric."""
        from indoorloc.evaluation import CDFAnalysis

        metric = CDFAnalysis(error_thresholds=[1, 5, 10, 20])
        result = metric.compute(self.predictions, self.ground_truths)

        assert isinstance(result, dict)
        assert 'within_1m' in result
        assert 'within_5m' in result
        assert 'within_10m' in result
        assert 'within_20m' in result

        # Higher thresholds should have higher percentages
        assert result['within_1m'] <= result['within_5m']
        assert result['within_5m'] <= result['within_10m']
        assert result['within_10m'] <= result['within_20m']


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_evaluator_default_metrics(self):
        """Test Evaluator with default metrics."""
        from indoorloc.evaluation import Evaluator
        from indoorloc.locations import Location

        predictions = [
            Location.from_coordinates(x=10, y=10, floor=0),
            Location.from_coordinates(x=20, y=20, floor=1),
        ]
        ground_truths = [
            Location.from_coordinates(x=12, y=12, floor=0),
            Location.from_coordinates(x=25, y=25, floor=1),
        ]

        evaluator = Evaluator()
        results = evaluator.evaluate(predictions, ground_truths)

        assert 'Mean Position Error (m)' in results
        assert 'Median Position Error (m)' in results
        assert 'Floor Accuracy (%)' in results

    def test_evaluator_custom_metrics(self):
        """Test Evaluator with custom metrics."""
        from indoorloc.evaluation import Evaluator, MeanPositionError, FloorAccuracy
        from indoorloc.locations import Location

        predictions = [
            Location.from_coordinates(x=10, y=10, floor=0),
        ]
        ground_truths = [
            Location.from_coordinates(x=10, y=10, floor=0),
        ]

        evaluator = Evaluator([MeanPositionError(), FloorAccuracy()])
        results = evaluator.evaluate(predictions, ground_truths)

        assert 'Mean Position Error (m)' in results
        assert 'Floor Accuracy (%)' in results
        assert len(results) == 2

    def test_evaluator_from_config(self):
        """Test building metrics from config dict."""
        from indoorloc.evaluation import Evaluator
        from indoorloc.locations import Location

        predictions = [
            Location.from_coordinates(x=10, y=10, floor=0),
        ]
        ground_truths = [
            Location.from_coordinates(x=12, y=12, floor=0),
        ]

        evaluator = Evaluator([
            {'type': 'MeanPositionError'},
            {'type': 'PercentileError', 'percentile': 90},
        ])

        results = evaluator.evaluate(predictions, ground_truths)
        assert 'Mean Position Error (m)' in results
        assert '90th Percentile Error (m)' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
