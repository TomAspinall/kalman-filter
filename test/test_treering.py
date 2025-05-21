from test._base import KalmanFilterBaseTest

test = KalmanFilterBaseTest("test/data/treering")

if __name__ == "__main__":
    test.run_test()
    test.test_ndim()
    test.test_log_likelihood()
    test.test_filtered_outputs()
    test.test_smoothed_outputs()

    expected = test.expected_filtered.to_dict()
    import kalman_filter as kf
    actual = kf.kalman_filter_verbose(test.test_KalmanFilter())

    # Output arrays need to be initially populated with na's, as they're curently random memory pointers.
    actual['vt']
    expected['vt']

    print("Treering test success!")
