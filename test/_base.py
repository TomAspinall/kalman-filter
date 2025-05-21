import json
import os

from numpy import float64

import kalman_filter as kf
from kalman_filter.base import BaseClassExtended


class KalmanFilterBaseTest(BaseClassExtended):
    """Base class providing an initial load of example measurements, which can then be used to derive a test suite for the example"""

    def __init__(self, measurements_path: str):
        """Load example measurements as a class attribute"""
        self.example_measurement_data = self._load_json(measurements_path)

    def _load_json(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path) as f:
            return json.load(f)


class KalmanFilterTest(KalmanFilterBaseTest):
    """Base class to build a test suite for the kalman-filter module.

    arg:
    - `expected_output_file`: A local path to a `.json` file of expected outputs, which can be coerced into a `KalmanFilterExpectedOutputs` object.
    - `example_measurements`: A local path to a `.json` file of example measurements, which are loaded as an available attribute: `example_measurement_data`
    - `absolute_threshold=float(1e-15)` the absolute threshold that results in an assertion success for various array checking between expected outputs and algorithm outputs.
    """

    def __init__(self, example_measurements_path_json: str, expected_output_path_json: str, absolute_threshold=float64(1e-15)):
        self.absolute_threshold = absolute_threshold
        super().__init__(example_measurements_path_json)

        # Load expected output data:
        self.expected_outputs = self._load_json(expected_output_path_json)

    # Base level tests:
    def assert_KalmanFilter(self):
        """Test that the `KalmanFilter` class object can be created successfully"""
        assert isinstance(kf.KalmanFilter(
            **self.kalman_filter_input), kf.KalmanFilter)
        assert isinstance(kf.KalmanFilter.from_dict(
            self.kalman_filter_input), kf.KalmanFilter)
        assert isinstance(kf.KalmanFilter(
            **self.kalman_filter_input).to_dict(), dict)
        return kf.KalmanFilter.from_dict(self.kalman_filter_input)

    def test_ndim(self):
        """Enforce expected dimensions of coerced input class object"""
        kalman_filter_obj = self.assert_KalmanFilter()
        for attribute, expected_ndim in kalman_filter_obj._attr_expected_ndims.items():
            assert expected_ndim == kalman_filter_obj.__getattribute__(
                attribute).ndim
