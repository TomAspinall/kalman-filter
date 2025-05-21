from functools import partial

import numpy as np

import kalman_filter as kf


class KalmanFilterBaseTest(kf.models.BaseClassExtended):
    """Base class to build a test suite for the kalman-filter module.

    arg:
    - `test_path`: A local path to a test directory. This directory expects 2 files: `filtere.json` and `smoothed.json`.
    - `absolute_threshold=float(1e-15)` the absolute threshold that results in an assertion success for various array checking between expected outputs and algorithm outputs.
    """

    def __init__(self, test_path: str, absolute_threshold=np.float64(1e-15)):
        self.test_path = test_path
        self.absolute_threshold = absolute_threshold

        # Object for performing filtering and smoothing:
        self.kalman_filter = kf.KalmanFilter.from_json(
            f"{test_path}/filtered.json")
        self.kalman_filter_input = self.kalman_filter.to_dict()

        # Expected output constants for comparison:
        self.expected_filtered = kf.KalmanFiltered.from_json(
            f"{test_path}/filtered.json")
        self.expected_smoothed = kf.KalmanFiltered.from_json(
            f"{test_path}/smoothed.json")

        # Threshold testing:
        self.is_close = partial(np.isclose, equal_nan=True,
                                atol=absolute_threshold)

    # Compare generated vs. expected dictionary outputs:
    def compare_attributes(self, actual: dict, expected: dict):
        # Compare filtered output against expected results:
        for expected_answer_key, expected_answer in expected.items():
            actual_answer = actual[expected_answer_key]
            # Is this an iterable?
            # Within threshold?
            outside_threshold = ~self.is_close(
                actual_answer, expected_answer)
            if hasattr(outside_threshold, "__len__"):
                try:
                    assert not outside_threshold.any()
                except AssertionError:
                    # Print total incorrect, and the first incorrect value:
                    total_incorrect = outside_threshold.sum()
                    incorect_elements = np.where(outside_threshold)
                    first_incorrect_index = tuple(
                        x[0] for x in incorect_elements)
                    first_incorrect_value = float(
                        expected_answer[first_incorrect_index])
                    first_expected_value = float(
                        actual_answer[first_incorrect_index])
                    incorrect_elements = ', '.join(
                        [str(x) for x in first_incorrect_index])
                    raise AssertionError(
                        f"{expected_answer_key} outside absolute tolerance!. {total_incorrect:,.0f} total failed elements. Element ({incorrect_elements}):  {first_incorrect_value:.04f} != {first_expected_value:.04f}")
            else:
                try:
                    assert not outside_threshold
                except AssertionError:
                    raise AssertionError(
                        f"{expected_answer_key} outside absolute tolerance! {expected_answer} != {actual_answer}")

    # Base level tests:
    def test_KalmanFilter(self):
        """Test that the `KalmanFilter` class object can be created successfully given algorithm inputs"""
        assert isinstance(kf.KalmanFilter(
            **self.kalman_filter_input), kf.KalmanFilter)
        assert isinstance(kf.KalmanFilter.from_dict(
            self.kalman_filter_input), kf.KalmanFilter)
        assert isinstance(kf.KalmanFilter(
            **self.kalman_filter_input).to_dict(), dict)
        return kf.KalmanFilter.from_dict(self.kalman_filter_input)

    def test_ndim(self):
        """Enforce expected dimensions of coerced input class object"""
        kalman_filter_obj = self.test_KalmanFilter()
        for attribute, expected_ndim in kalman_filter_obj._attr_expected_ndims.items():
            assert expected_ndim == kalman_filter_obj.__getattribute__(
                attribute).ndim

    def test_log_likelihood(self) -> None:
        kalman_filter_obj = self.test_KalmanFilter()
        # Calculate log-likelihood:
        log_likelihood = kf.kalman_filter(kalman_filter_obj)
        assert self.is_close(
            log_likelihood, self.expected_filtered.log_likelihood)

    def __repr__(self) -> str:
        return f"KalmanFilterBaseTest(test_path='{self.test_path}', absolute_threshold={self.absolute_threshold})"

    def test_filtered_outputs(self):
        # Test class object construction:
        kf_object = self.test_KalmanFilter()

        # Calculate filtered output:
        filtered = kf.kalman_filter_verbose(kf_object)

        self.compare_attributes(
            filtered.to_dict(), self.expected_filtered.to_dict())

    def test_smoothed_outputs(self):
        # Test class object construction:
        kf_object = self.test_KalmanFilter()

        # Calculate filtered output:
        smoothed = kf.kalman_smoother(kf_object)

        self.compare_attributes(
            smoothed.to_dict(), self.expected_smoothed.to_dict())

    def run_test(self):
        self.test_ndim()
        self.test_log_likelihood()
        self.test_filtered_outputs()
        self.test_smoothed_outputs()
